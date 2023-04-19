# coding=utf-8
# =============================================
# @Time      : 2023-04-19 17:32
# @Author    : DongWei1998
# @FileName  : train_Unigram.py
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
from tokenizers import Regex

tokenizer = Tokenizer(models.Unigram())



tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]

trainer = trainers.UnigramTrainer(
    vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
)

# 获取训练数据
with open('sen_list.txt', 'r', encoding='utf-8') as r:
    dataset = r.readlines()
# 数据迭代器
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]


tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()


tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", tokenizer.token_to_id("<sep>")), ("<cls>", tokenizer.token_to_id("<cls>"))],
)


tokenizer.decoder = decoders.Metaspace()


# 模型保存
tokenizer.save("tokenizer_Unigram.json")

# 模型使用
new_tokenizer = Tokenizer.from_file("tokenizer_Unigram.json")


from transformers import XLNetTokenizerFast

wrapped_tokenizer = XLNetTokenizerFast(tokenizer_object=new_tokenizer)
wrapped_tokenizer.save_pretrained("../Unigram")