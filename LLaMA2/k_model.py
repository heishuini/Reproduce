from transformers import PretrainedConfig
from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch import nn
import torch
import torch.nn.functional as F

import math


# 超参数设置
class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768, # 模型维度
            n_layers: int = 12, # Transformer的层数
            n_heads: int = 16, # 注意力机制的头数
            n_kv_heads: int = 8, # 键值头的数量
            vocab_size: int = 6144, # 词汇表大小
            hidden_dim: int = None, # 隐藏层维度
            multiple_of: int = 64, 
            norm_eps: float = 1e-5, # 归一化层的eps
            max_seq_len: int = 512, # 最大序列长度
            dropout: float = 0.0, # dropout概率
            flash_attn: bool = True, # 是否使用Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

# 归一化层
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps防止除以0
        self.eps = eps
        # 可学习参数，初始化为全1
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        # 计算RMSNorm的核心部分
        # x.pow(2).mean(-1, keepdim=True)计算了输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # forward函数是模型的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
# 测试RMSNorm
# norm = RMSNorm(dim=768, eps=1e-5)
# x = torch.randn(1, 50, 768)  # 创建一个随机张量
# output = norm(x)  # 应用RMSNorm
# print(output.shape)  # 输出形状应该与输入相同 [1, 50, 768]

## 分组注意力GQA
# 将键K和值V的维度扩展至与查询Q的维度一样
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim= x.shape
    
    # 重复次数为1，则不需要重复，返回原始张量
    if n_rep == 1:
        return x
    
    # 对张量进行扩展和重塑以重复键值对
    return (
        x[:, :, :, None, :] # 在第四个维度(头维度前)添加一个新维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim) # 将新添加的维度扩展到n_rep大小，实现重复的效果
        .reshape(bs, slen, n_kv_heads*n_rep, head_dim) # 重新塑形，合并键/值对头的数量和重复次数的维度
    )

# 旋转嵌入，加强attention的上下文信息
# 获取旋转嵌入的实部cos和虚部sin
# 此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arrange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arrange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

# 调整freqs_cis的形状，使其广播操作时与x的维度对齐，以正确进行张量运算
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    # 确保1在x的维度范围内
    assert 0 <= 1 < ndim
    # 确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim-1 else 1 for i, d in enumerate(x.shape)]

    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)

# 实现旋转嵌入，需要拆分为实部和虚部，然后做运算，最后拼接合并回来
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    # reshape: 将最后一个维度 head_dim 拆分为 (head_dim // 2, 2)，例如 128 维变为 (64, 2)。
    # unbind(-1) 将最后一个维度分离成两个张量
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin # 实部旋转 rcos-isin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos # 虚部旋转 rsin+icos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，还原为原始张量的形状
    # stack:将实部和虚部重新拼接为 (..., head_dim//2, 2)
    # flatten(3): 合并最后两个维度，恢复原始形状 (batch, seq_len, n_heads, head_dim)。
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
# 旋转emb测试
# xq = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
# xk = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
# # 使用 precompute_freqs_cis 函数获取 sin和cos
# cos, sin = precompute_freqs_cis(288//6, 50)
# print(cos.shape, sin.shape)
# torch.Size([50, 24]) torch.Size([50, 24])

# xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)
# xq_out.shape, xk_out.shape
# (torch.Size([1, 50, 6, 48]), torch.Size([1, 50, 6, 48]))

# Attention
class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # 根据是否指定n_kv_heads，确定用于键（key）和值（value）的头的数量。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数被键值头数整除
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小，默认为1
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值总头数除以模型并行处理大小
        self.n_local_kv_heads = args.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸以匹配查询Q的头数尺寸
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以总头数
        self.head_dim = args.dim // args.n_heads

        # 定义映射的权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        # 输出的权重矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # dropout概率
        self.dropout = args.dropout
        
        # 检查是否使用Flash Attention（需要PyTorch >= 2.0）。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 不支持Flash Attention，采用手动实现的注意力机制，并设置mask
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，屏蔽未来信息
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # 计算Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应头的维度
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE，旋转嵌入
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 分组查询注意力，GQA，多设计了kv头，需要对键和值进行扩展以适应重复次数
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理
        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)

        # 根据是否支持Flash Attention，选择实现方式。
        if self.flash:
            # 使用Flash Attention
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 手动实现的注意力
            # 计算qk^T / dim
            scores = torch.matmul(xq, xk.transpose(2,3)) / math.sqrt(self.head_dim)
            assert hasattr(self,'mask')
            # 加上mask
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            # softmax转为权重得分
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # 做dropout            
            scores = self.attn_dropout(scores)
            # 乘以v分配权重
            output = torch.matmul(scores, xv)

        # 恢复时间维度合并头
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # 投影回残差
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output
# attention测试
# 创建Attention实例
attention_model = Attention(args)

# 模拟输入数据
# batch_size = 1
# seq_len = 50  # 假设实际使用的序列长度为50
# dim = args.dim
# x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
# # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
# # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

# freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

# # 运行Attention模型
# output = attention_model(x, freqs_cos, freqs_sin)

# # attention出来之后的形状 依然是[batch_size, seq_len, dim]
# print("Output shape:", output.shape)
# Output shape: torch.Size([1, 50, 768])

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3) # 减少开销
            hidden_dim = multiple_of *((hidden_dim + multiple_of - 1) // multiple_of) # 硬件对齐

        # 定义第一层线性变换，输入->隐藏
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以 输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# MLP测试
# 创建MLP实例
# mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
# # 随机生成数据
# x = torch.randn(1, 50, args.dim)
# # 运行MLP模型
# output = mlp(x)
# print(output.shape)
# torch.Size([1, 50, 768])

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # 多头注意力的头数
        self.n_heads = args.n_heads
        # 输入维度
        self.dim = args.dim
        # 每个头的维度
        self.head_dim = args.dim // args.n_heads
        # 注意力操作
        self.attention = Attention(args)
        # MLP
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 层的ID
        self.layer_id = layer_id
        # 归一化层(注意力计算的，MLP的)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x, freqs_cos, freqs_sin):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        # freqs_cos和freqs_sin用于RoPE
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

# Decoder测试
# 创建LLaMADecoderLayer实例
# decoderlayer = DecoderLayer(0, args)

# # 模拟输入数据
# dim = args.dim
# seq_len = 50

# x = torch.randn(1, seq_len, dim) # [bs, seq_len, dim]

# freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

# out = decoderlayer(x, freqs_cos, freqs_sin)

# print(out.shape) # 形状和输入的x一样 [batch_size, seq_len, dim]
# torch.Size([1, 50, 768])


class Transformer(PreTrainedModel):
    config_class = ModelConfig # 配置类
    last_loss: Optional[torch.Tensor] # 记录最后一次计算的损失

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层权重与输出层权重共享
        self.tok_embeddings.weight = self.output.weight

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        #  persistent=False,这些缓冲区不会保存在模型的状态字典（state_dict）中
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast() # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()] # 不分割的模块列表

    def _init_weights(self, module):
        # 线性层投影
        # 权重normal，bias为0
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # 嵌入层权重
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **keyargs) -> torch.Tensor:
        """
            - tokens: Optional[torch.Tensor], 输入 token 张量。
            - targets: Optional[torch.Tensor], 目标 token 张量。
            - kv_cache: bool, 是否使用键值缓存。
            - keyargs: 其他关键字参数。

            - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        """
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']

        # 前向传播
        _bsz, seqlen = tokens.shape

        # 结果词嵌入和dropout
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        # 获取相对位置嵌入的频率，用于RoPE
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过decoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定目标token，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            # 推理时小优化：只对最后一个位置的输出进行前向传播
            logits = self.output(h[:, [-1], :])
            self.last_loss = None
        
        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)

        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        idx一般是输入文本经过tokenizer化后的id序列
        """
        
        index = idx.shape[1] # 记录起始位置
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            # 当序列超长时，只保留最后 max_seq_len 个 token（滑动窗口机制）
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # Logits：模型最后一层（全连接层）的直接输出
            # 前向传播获取序列中最后一个位置的logits
            logits = self(idx_cond).logits # 全序列预测
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出

            # 确定性生成，选取概率最高的token
            if temperature == 0.0:
                # 选取最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            # 随机性生成
            else:
                # 缩放logits并应用softmax
                logits = logits / temperature
                # topk筛选
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # 转为概率分布
                probs = F.softmax(logits, dim=-1)
                # 概率采样
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)
        
        # 只返回生成的token
        return idx[:, index:]

# Transformer模块测试
# LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
# x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
# # 实例化LLaMA2Model
# model = Transformer(args=args)
# # 计算model的全部参数
# num_params = sum(p.numel() for p in model.parameters())
# print('Number of parameters:', num_params)
# Number of parameters: 82594560

# out = model(x)
# print(out.logits.shape) # [batch_size, 1, vocab_size]
# torch.Size([1, 1, 6144])


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_k")
    args = ModelConfig(
        dim=1024,
        n_layers=18,
    )
    # 实例化LLaMA2Model
    model = Transformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print(f'LLM总参数量:{num_params / 1e6:.3f} 百万')

    prompt = "你好呀，今天吃什么呢？你过得怎么样嘞？"
    text = f"{tokenizer.bos_token}{prompt}{tokenizer.eos_token}"
    print(f"Input text: {text}")

    input_id = tokenizer(text).data['input_ids']
    print("input_ids :", input_id)
    print("dcode_str :", tokenizer.decode(input_id))

    X = torch.tensor(input_id[:-1]).unsqueeze(0)
    Y = torch.tensor(input_id[1:]).unsqueeze(0)
    print("X shape :", X.shape)
    print("Y shape :", Y.shape)

    # 将输入张量传入模型
    output = model(X, Y)
