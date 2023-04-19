# coding=utf-8
# =============================================
# @Time      : 2023-04-19 16:31
# @Author    : DongWei1998
# @FileName  : token_test.py
# @Software  : PyCharm
# =============================================

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('/home/20230416/roberta/WordPiece')
questions =  ["我当 年汝 南有 个善于 评论人 物的 名士","当年汝南有个阿斯顿善于评论人物的名士","当年汝南有个善于当时法国使得否阿斯蒂芬阿斯顿评论人物的名点发噶点发噶点发噶士"]
examples = ["我当年汝南有个善于评论人物的名士","当年汝南有个阿斯顿善于评论人物的名士","当年汝南有个善于当时法国使得否阿斯蒂芬阿斯顿评论人物的名点发噶点发噶点发噶士"]
inputs = tokenizer(
    questions,
    examples,
    max_length=384,
    truncation="only_second",
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
print(inputs)



from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('/home/20230416/roberta/BPE')
questions =  ["我当 年汝 南有 个善于 评论人 物的 名士","当年汝南有个阿斯顿善于评论人物的名士","当年汝南有个善于当时法国使得否阿斯蒂芬阿斯顿评论人物的名点发噶点发噶点发噶士"]
examples = ["我当年汝南有个善于评论人物的名士","当年汝南有个阿斯顿善于评论人物的名士","当年汝南有个善于当时法国使得否阿斯蒂芬阿斯顿评论人物的名点发噶点发噶点发噶士"]
inputs = tokenizer(
    questions,
    examples,
    max_length=384,
    truncation="only_second",
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
print(inputs)






from transformers import XLNetTokenizerFast
tokenizer = XLNetTokenizerFast.from_pretrained('/home/20230416/roberta/Unigram')
questions =  ["我当 年汝 南有 个善于 评论人 物的 名士","当年汝南有个阿斯顿善于评论人物的名士","当年汝南有个善于当时法国使得否阿斯蒂芬阿斯顿评论人物的名点发噶点发噶点发噶士"]
examples = ["我当年汝南有个善于评论人物的名士","当年汝南有个阿斯顿善于评论人物的名士","当年汝南有个善于当时法国使得否阿斯蒂芬阿斯顿评论人物的名点发噶点发噶点发噶士"]
inputs = tokenizer(
    questions,
    examples,
    max_length=384,
    truncation="only_second",
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
print(inputs)


