import torch
import torch.nn as nn

'''
  对样本的特征layernorm, 每个位置,
  所以需要传入特征数量d_model
'''
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps # 防止除以0
    
    def forward(self, x):
        # '-1' means last dimension.
        # -1：表示在最后一个维度上计算均值。例如，如果 x 的形状是 (batch_size, seq_len, features)，这会在特征维度（features）上计算均值。
        # keepdim=True：保持输出的维度与输入一致。如果不设置 keepdim=True，计算均值后的形状会是 (batch_size, seq_len)（去掉最后一个维度）
        mean = x.mean(-1, keepdim=True) #  形状: (batch_size, seq_len, 1)
        
        # unbiased=False, 表示计算方差时使用 有偏估计（biased variance），即分母是 n（样本数）。
        # 如果 unbiased=True，则分母是 n-1（无偏估计）。
        var = x.var(-1, unbiased=False, keepdim=True) # # 形状: (batch_size, seq_len, 1)
        
        # 将原始数据变成均值为0，方差为1的分布，便于训练，防止落在激活函数敏感区而梯度爆炸/消失
        out = (x - mean) / torch.sqrt(var + self.eps) # 形状: (batch_size, seq_len, d_model)
        
        # 缩放偏移 保留模型的表达能力，缓解归一化造成的信息丢失
        out = self.gamma * out + self.beta
        
        return out
        
        
        