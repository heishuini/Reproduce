import torch.nn as nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention

'''
    params: 投影维度d_model, 分头数目n_head
'''
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model,
                 n_head
    ):
        super().__init__()
        
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        
        # 多头注意力，首先进行投影，此处没有选择投影至低维
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    
    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        
        # 拆分为多个头，如512/64 = 8，64个头，每个头维度是8
        d_tensor = d_model // self.n_head 
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)
        
        return tensor
    
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        
        # contigous()确保张量的内存布局连续，符合view要求
        # 把head和d_tensor合并
        tensor = tensor.transpose(1, 2).contigous().view(batch_size, length, d_model)
        return tensor
    
    def forward(self, q, k, v, mask=None):
        '''
            [batch_size, length, d_model] 
            -> split: [batch_size, n_head, length, d_tensor], d_tensor = d_model // n_head
            -> sdpa(q,k,v)
            -> concat: [batch_size, length, d_model]
        '''
        # 1. project
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. split head, 将d_model拆成head和d_tensor扩展维度。
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        # 3. scale dot, 返回v和score
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat & linear
        out = self.concat(out)
        out = self.w_concat(out)
        
        # todo: 5. visualize score
        
        return out