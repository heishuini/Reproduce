import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # 用GELU作为激活函数， transformer是relu，这是借鉴BERT了。
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 拆为多个头
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # softmax
        attn = self.attend(dots)
        
        # 然后dropout?
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        # 合并头
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 输出投影
        return self.to_out(out)


class Transformer(nn.Module):
    '''
    transformer encoder
    '''
    def __init__(self, 
                 dim, 
                 depth, # 块数
                 heads, 
                 dim_head, 
                 mlp_dim, 
                 dropout = 0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 多头attention
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        # x shape: (b, n, d)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, 
                 *, 
                 image_size, 
                 patch_size, 
                 num_classes, 
                 dim,  # embedding_dim
                 depth, 
                 heads, 
                 mlp_dim, 
                 pool = 'cls', 
                 channels = 3, 
                 dim_head = 64, 
                 dropout = 0., 
                 emb_dropout = 0.
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 一个patch块的像素量
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # patch embedding
        self.to_patch_embedding = nn.Sequential(
            # 输入为(b, c, H, W),拆为如下形式(b, n, d)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),  # transformer 一般用的是LN
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # (num_patches, dim) add
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # (1, dim) concat
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # 分类头
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # concat cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # add pos embedding
        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)
        
        # 进入transformer-encoder
        x = self.transformer(x)

        # 取cls token，第0个
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # identity
        x = self.to_latent(x)
        
        # 分类头
        return self.mlp_head(x)
    
    
import torch
# pip install vit-pytorch
from vit_pytorch import ViT

if __name__ == "__main__":

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img) # (1, 1000)