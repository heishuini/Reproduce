import torch.nn as nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 n_head, 
                 drop_prob
    ):
        # super().__init__()  python3 不需要传递类名和实例，Python 会自动推断
        super(EncoderLayer,self).__init__() # python2 显式指定当前类名 (EncoderLayer) 和实例 (self)
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, src_mask):
        # 1. self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        # 如果先 Add（残差连接）再 Dropout，残差路径（_x）的信息会被随机丢弃，这会破坏残差连接的核心思想
        # 原始 Transformer 的顺序是： Add → Norm → Dropout 但后续研究发现，先 Dropout 再 Add & Norm 更稳定，尤其是在深层网络中。
        x = self.dropout1(x) # 先dropout
        x = self.norm1(x + _x)
        
        # 3. ffn
        _x = x
        x = self.ffn(x)
        
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        
        return x
        