# coding=utf-8
# =============================================
# @Time      : 2023-04-19 16:51
# @Author    : DongWei1998
# @FileName  : train_BPE.py
# @Software  : PyCharm
# =============================================
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.BPE())


pre_tokenizer = pre_tokenizers.WhitespaceSplit()

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)


print(tokenizer.pre_tokenizer.pre_tokenize_str("十月革命胜利,世界上出现了第一个社会主义国家"))


# 获取训练数据
with open('sen_list.txt', 'r', encoding='utf-8') as r:
    dataset = r.readlines()


# 数据迭代器
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]

trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>",'[PAD]'])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)


tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)


tokenizer.decoder = decoders.ByteLevel()



encoding = tokenizer.encode("十月革命胜利,世界上出现了第一个社会主义国家")
print(encoding.tokens)


print(tokenizer.decode(encoding.ids))


# 模型保存
tokenizer.save("tokenizer_BPE.json")

# 模型使用
new_tokenizer = Tokenizer.from_file("tokenizer_BPE.json")

from transformers import GPT2TokenizerFast
wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=new_tokenizer)
wrapped_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
wrapped_tokenizer.save_pretrained("../BPE")


