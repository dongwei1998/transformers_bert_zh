#  BPE 字节对编码
字节对编码最初被开发为一种压缩文本的算法,然后在预训练 GPT 模型时被 OpenAI 用于标记化。许多 Transformer 模型都使用它,
包括 GPT、GPT-2、RoBERTa、BART 和 DeBERTa。


# WordPiece 
是 Google 为预训练 BERT 而开发的标记化算法。此后,它在不少基于 BERT 的 Transformer 模型中得到重用
例如 DistilBERT、MobileBERT、Funnel Transformers 和 MPNET


# SentencePiece 
中经常使用 Unigram 算法 该算法是 AlBERT、T5、mBART、Big Bird 和 XLNet 等模型使用的标记化算法。