import torch.nn as nn
import torch.nn.functional as F
import torch

import math

class Attention(nn.Module):
    """
    compute Scale Dot Product Attention (SDPA)
    """
    def __init__(self):
        super().__init__()
        
        
    def forward(self, q, k, v, mask=None, dropout=None):
        # [batch, num_head, sequence, dim]
        # 点乘
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算得分
        p_attn = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
            
        return torch.matmul(p_attn, v), p_attn