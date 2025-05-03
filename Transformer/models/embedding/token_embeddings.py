import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    本质是一个可学习的查找表（lookup table），将整数 token 索引映射为稠密向量。
    """
    def __init__(self, vocab_size, d_model):
        '''
        vocab_size: 词汇表的大小（唯一 token 的数量）。 行数
        d_model: 词嵌入的维度（每个 token 用多少维的向量表示）。 列数
        padding_idx: 不同句子的长度可能不同，通常会用 padding 填充到统一长度。
                     索引 1 的 token 会被视为无效位置，其嵌入向量固定为全零，且不参与梯度更新
        '''
        super().__init__(vocab_size, d_model, padding_idx=1)