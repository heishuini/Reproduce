import torch.nn as nn
import math

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        
        # 1. 计算分数
        k_t = k.transpose(2,3)
        score = (q @ k_t) / math.sqrt(d_tensor) # scale
        
        # 2. mask(opt)
        if mask is not None:
            # masked_fill(mask, value), mask是布尔型张量，将对应True地方替换为value
            score = score.masked_fill(mask == 0, -10000)
        
        # 3. softmax, 最后一个维度(length维度)进行softmax元素计算
        score = self.softmax(score)
        
        # 4. x value
        v = score @ v
        
        return v, score