import torch.nn as nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 n_head, 
                 drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
        
    def forward(self, dec, enc, trg_mask, src_mask):
        '''
        dec: decoder输入
        enc: encoder输出
        trg_mask: decoder_mask
        src_mask:  encoder_mask
        '''
        # 1. self-attention (mask)
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. encoder - decoder attention
        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)
            
        # 5. ffn
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        
        return x