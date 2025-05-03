from torch import nn, optim
import torch
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
import time
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


### ---- conf ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model parameter setting
batch_size = 128
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1
# optimizer parameter setting
init_lr = 1e-5
factor = 0.9 # 学习率衰减因子
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0 # 限制梯度最大值
weight_decay = 5e-4
inf = float('inf')

### ---- data ----
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'), # 英语为源语言，德语为目标语言
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>') # <sos>(序列开始)和<eos>(序列结束)

train, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train, min_freq=2) # 基于训练数据构建源语言和目标语言的词汇表
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)
# 获取关键token的索引，供模型使用
src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)

# ---- model -----
from models.model.transformer import Transformer
from torch.optim import Adam

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
# 初始化
model.apply(initialize_weights)
# 优化器
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

# 学习率调度器，根据传入的验证指标调整
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, # 要优化的优化器对象
                                                 verbose=True, # 是否打印调整信息
                                                 factor=factor, # 学习率衰减因子，lr*factor
                                                 patience=patience)  # 等待轮数
# 忽略pad索引所在位置的计算
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src # 源语言序列 [batch_size, src_len]
        trg = batch.trg # 目标语言序列 [batch_size, trg_len]

        optimizer.zero_grad()
        output = model(src, trg[:, :-1]) # [batch_size, trg_len-1, output_dim]
        
        output_reshape = output.contiguous().view(-1, output.shape[-1]) # [(batch_size*(trg_len-1)), output_dim]
        trg = trg[:, 1:].contiguous().view(-1)  # [(batch_size*(trg_len-1))]

        loss = criterion(output_reshape, trg)
        
        loss.backward()
        
         # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval() # 关闭dropout和batch normalization的随机性
    epoch_loss = 0
    batch_bleu = []
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            # 计算BLEU分数
            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        # 训练
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        # 验证
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        # 前面warmup阶段，让模型稳定一些然后再更新学习率
        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)