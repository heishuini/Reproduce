# Pylance诊断范围改为workspace
from models.model.encoder import Encoder
from models.model.decoder import Decoder
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 trg_sos_idx, 
                 enc_voc_size, 
                 dec_voc_size, 
                 d_model, 
                 n_head, 
                 max_len,
                 ffn_hidden, 
                 n_layers, 
                 drop_prob, 
                 device
    ):
        super().__init__()
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
    
    '''
    src: 源序列（如待翻译的原始句子），形状为 (batch_size, src_seq_len)
    trg: 目标序列（如翻译结果），形状为 (batch_size, trg_seq_len)
    '''
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) # 原序列掩码
        trg_mask = self.make_trg_mask(trg) # 目标序列掩码
        
        enc_src = self.encoder(src, src_mask) # 编码
        output = self.decoder(trg, enc_src, trg_mask, src_mask) # 解码
        return output
    
    # 源
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # 是掩码的位置则置False
        # 从 (batch_size, seq_len) 扩展为 (batch_size, 1, 1, seq_len)，以便广播到多头注意力的所有头。
        return src_mask
    
    # 目标
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3) # (batch_size, 1, trg_seq_len, 1)
        trg_len = trg.shape[1]
        
        # torch.tril取下三角包括对角线
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        #   [[[ True, False, False],  # 第1个词只能看自己
        #     [ True,  True, False],  # 第2个词可看前两个
        #     [ True,  True, False]]] # 第3个词（填充符）无效
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return trg_mask