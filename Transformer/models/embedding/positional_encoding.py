import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super().__init__()
        
        # same size with input matrix (for adding with input matrix)
        # 创建形状为[max_len, d_model]的零矩阵
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        
        # 创建位置序列[0, 1, 2, ..., max_len-1]
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1) # 从1D变为2D [max_len, 1]
        
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        # 创建维度索引序列，步长为2 [0, 2, 4, ..., d_model//2*2]
        _2i = torch.arrange(0, d_model, step=2, device=device).float()
        
        # compute positional encoding to consider positional information of words
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        
        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]
        
        # 返回对应位置编码
        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
        #  位置0: [sin(0), cos(0), sin(0/10000^(2/4)), cos(0/10000^(2/4))]
        #  位置1: [sin(1), cos(1), sin(1/10000^(2/4)), cos(1/10000^(2/4))]