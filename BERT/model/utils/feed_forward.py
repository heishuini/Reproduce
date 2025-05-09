import torch.nn as nn
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # 区别点：用的是GELU函数
        self.activation = GELU()

    def forward(self, x):
        # 投影 -> 激活 -> dropout -> 投影
        return self.w_2(self.dropout(self.activation(self.w_1(x))))