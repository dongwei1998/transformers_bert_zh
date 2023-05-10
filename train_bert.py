# coding=utf-8
# =============================================
# @Time      : 2023-04-01 11:40
# @Author    : DongWei1998
# @FileName  : train_roberta.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用
os.environ['HF_HOME'] = './datasets_str'
from transformers import BertTokenizerFast
from transformers import TFAutoModelForQuestionAnswering
from datasets import load_dataset  # noqa: F401
from datasets import load_metric
from transformers import create_optimizer
import tensorflow as tf
import collections
import numpy as np
from transformers import DefaultDataCollator
from tqdm.auto import tqdm
from huggingface_hub import notebook_login
# 登录huggingface
notebook_login()
# from huggingface_hub import login
# login()

# 超参数
max_length = 384
stride = 128
n_best = 20
max_answer_length = 30
num_train_epochs = 1


tok = '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'

# 文本向量化工具
model_checkpoint = "/home/20230416/roberta/bert-finetuned-squad_zh"
if not os.path.exists(model_checkpoint):
    model = TFAutoModelForQuestionAnswering.from_pretrained('huggingface-course/bert-finetuned-squad')
    tokenizer = BertTokenizerFast.from_pretrained('/home/20230416/roberta/my_new_tokenizer')
else:
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)

# 数据加载
data_files = {"train": "datasets_str/qa_me_train_1.json","validation":"datasets_str/qa_me_validation_1.json"}
raw_datasets = load_dataset("json", data_files=data_files)

# 数据展示
print(raw_datasets)
print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])
print(raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1))
print(raw_datasets["validation"].filter(lambda x: len(x["answers"]["text"]) != 1))



# 训练集数据处理
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


train_dataset = raw_datasets["train"].select(range(10)).map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
# train_dataset = raw_datasets["validation"]
print(len(raw_datasets["train"]))

# 验证集数据处理
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(sample_idx)

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

validation_dataset = raw_datasets["validation"].select(range(10)).map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

print(len(raw_datasets["validation"]))



metric = load_metric("squad")



# 数据转换

data_collator = DefaultDataCollator(return_tensors="tf")
tf_train_dataset = train_dataset.to_tf_dataset(
    columns=[
        "input_ids",
        "start_positions",
        "end_positions",
        "attention_mask",
    ],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=4,
)

tf_eval_dataset = validation_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=4,
)



def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    return metric.compute(predictions=predicted_answers, references=theoretical_answers)



# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_train_steps = len(tf_train_dataset) * num_train_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")





from transformers.keras_callbacks import PushToHubCallback
callback = PushToHubCallback(output_dir="bert-finetuned-squad_zh", tokenizer=tokenizer)
model.fit(tf_train_dataset, callbacks=[callback],epochs=num_train_epochs)


# # 模型评估
# # 文本向量化工具
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
predictions = model.predict(tf_eval_dataset)
res = compute_metrics(
    predictions["start_logits"],
    predictions["end_logits"],
    validation_dataset,
    raw_datasets["validation"],
)
print(res)



