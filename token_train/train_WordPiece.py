# coding=utf-8
# =============================================
# @Time      : 2023-04-19 14:22
# @Author    : DongWei1998
# @FileName  : train_WordPiece.py
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
#
# #
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# 规范器
# tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

# Whitespace 预标记器会在空格和所有非字母、数字或下划线字符的字符上进行拆分
# pre_tokenizer = pre_tokenizers.Whitespace()
# print(pre_tokenizer.pre_tokenize_str("中国 在 那里?"))

# WhitespaceSplit()只在空白处进行拆分
pre_tokenizer = pre_tokenizers.WhitespaceSplit()
print(pre_tokenizer.pre_tokenize_str("中国 在 那里 ?"))


special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

# 获取训练数据
with open('sen_list.txt', 'r', encoding='utf-8') as r:
    dataset = r.readlines()
# 数据迭代器
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]

# 训练模型
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)


# 构建模板
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
)
# 编码
tokenizer.enable_padding()
# 截断
tokenizer.enable_truncation(max_length=384,stride=128)

encoding = tokenizer.encode("我当年汝南有个善于评论人物的名士")
print(encoding.tokens)

# 解码器
tokenizer.decoder = decoders.WordPiece(prefix="##")
tokenizer.decode(encoding.ids)

# 模型保存
tokenizer.save("tokenizer_WordPiece.json")



# 模型使用
new_tokenizer = Tokenizer.from_file("tokenizer_WordPiece.json")
from transformers import BertTokenizerFast
wrapped_tokenizer = BertTokenizerFast(tokenizer_object=new_tokenizer)
wrapped_tokenizer.save_pretrained("../WordPiece")
