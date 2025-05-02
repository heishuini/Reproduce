import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, 
                 dec_voc_size, 
                 max_len, 
                 d_model, 
                 ffn_hidden, 
                 n_head, 
                 n_layers, 
                 drop_prob, 
                 device
    ):
        super().__init__()
        