import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torchvision

os.makedirs("images", exist_ok=True)

# ---- settings
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epoches of training")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--img_size", type=int, default=28)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # linear -> normalize -> activate
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                # 1d:支持2D（batch_size, num_features）或3D（batch_size, num_features, sequence_length）的输入
                # 2d: 仅支持4D输入（batch_size, num_channels, height, width）
                layers.append(nn.BatchNorm1d(out_feat, eps=0.8))
            #  ReLU（Rectified Linear Unit）的改进版本
            # 输入小于0时，negative_slope * x； 大于等于0，输出x
            # 保留负值区域的微小梯度，缓解神经元死亡问题
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return layers
    
        self.model = nn.Sequential(
            # *表示解包，将bloock的layers拆开叠加
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            # prod计算总像素， 映射至像素空间
            nn.Linear(1024, int(np.prod(img_shape))),
            # 双曲正切，[-1,1]
            # ReLU 不限制输出范围，可能导致像素值爆炸，不适合直接生成图像。
            # Sigmoid 输出 [0, 1]，但可能导致梯度饱和（两端梯度接近零）。
            nn.Tanh()
        )
        
    def forward(self, z):
        # 接收noise为输入，输出a batch of imgs
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        
        return validity
    
# loss function
# BCELoss: 二分类任务损失函数 - ylog(p)+(1-y)log(1-p)
adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

if cuda:
    # 将parameters全部移到GPU
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    
# data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1,opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 初始化 TensorBoard
writer = SummaryWriter(log_dir="logs/gan_training_epoch200")


# ----- Training
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # adversarial gt
        # 在旧版PyTorch（<0.4）中，Variable 是对张量的封装，用于自动求导。
        
        # 真实标签和假标签
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=True)
        
        # input
        real_imgs = Variable(imgs.type(Tensor))
        
        # --- train Generator
        optimizer_G.zero_grad()
        
        # sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        
        # generate a batch of images
        gen_imgs = generator(z)
        
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid) # 希望discriminator判别结果是真实
        
        g_loss.backward()
        optimizer_G.step()
        
        # --- train Discriminator
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        # 希望discriminator能区别real和fake，其loss最好是二者相加/2
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        # .detach()会创建一个新的张量，该张量不保留计算历史，且requires_grad=False
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2 # 最优
        
        d_loss.backward()
        optimizer_D.step()
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        
        batches_done = epoch * len(dataloader) + i
        
        # 记录损失值
        writer.add_scalar("Loss/Discriminator", d_loss.item(), batches_done)
        writer.add_scalar("Loss/Generator", g_loss.item(), batches_done)
        
        # if batches_done % opt.sample_interval == 0:
            # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            
        if batches_done % opt.sample_interval == 0:
            img_grid = torchvision.utils.make_grid(gen_imgs.data[:25], nrow=5, normalize=True)
            writer.add_image("Generated Images", img_grid, batches_done)
            
# 关闭写入器
writer.close()
         