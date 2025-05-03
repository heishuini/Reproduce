# from torchtext.legacy.data import Field, BucketIterator
# from torchtext.legacy.datasets.translation import Multi30k

from torchtext.data import Field, BucketIterator
from torchtext.datasets.translation import Multi30k

# 加载和预处理双语（德语-英语）机器翻译数据集（如Multi30k）
class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext # 文件扩展名元组，如('.de', '.en')
        self.tokenize_en = tokenize_en # 英语分词函数
        self.tokenize_de = tokenize_de # 德语分词函数
        self.init_token = init_token  # 句子起始token
        self.eos_token = eos_token # 句子结束token
        print('dataset initializing start')

    def make_dataset(self):
        # 根据文件扩展名设置源语言和目标语言的Field
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
        # 加载Multi30k数据集
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    # 基于训练数据构建词汇表
    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    # 创建批数据迭代器
    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator