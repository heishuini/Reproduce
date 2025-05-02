import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 n_head, 
                 drop_prob
    ):
        # super().__init__()  python3 不需要传递类名和实例，Python 会自动推断
        super(EncoderLayer,self).__init__() # python2 显式指定当前类名 (EncoderLayer) 和实例 (self)
        