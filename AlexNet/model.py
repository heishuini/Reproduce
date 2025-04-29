"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # 输入图像大小
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
# DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
DEVICE_IDS = [0] 
# modify this to point to your data directory
INPUT_ROOT_DIR = 'data'
TRAIN_IMG_DIR = 'data/imageNet'
OUTPUT_DIR = 'output'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            # ---- 第一层卷积层 ----
            # 1. 卷积 (B, 3, 227, 227) -> (B, 96, 55, 55) h = (h+2p-k)/s + 1 = (227+0-11)/4 + 1 = 55  
            # 如果是224输入，算出来是54.25
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            # 2. 激活 max(0,x)
            nn.ReLU(), 
            # 3. 归一化操作, 局部归一化(亮度归一化) local contrast normalization, brightness normalization
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            # 4. 最大值池化 (B, 96, 55, 55) -> (B, 96, 27, 27), h = (55+0-3)/2 + 1 = 27
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)

            # ---- 第二层卷积层 ----  bias = 1 
            # 在 PyTorch 中，nn.Conv2d 的偏置参数 bias 是一个一维张量，其长度等于输出通道数
            # (27 +4 -5)/1 + 1 = 27
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)

            # ---- 第三层卷积层 ----
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),

            # ---- 第四层卷积层 ----  bias = 1
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),

            # ---- 第五层卷积层 ----  bias = 1
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )

        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            # ---- 第一层全连接层 ----
            nn.Dropout(p=0.5, inplace=True),
            # Linear, 最后一个维度对齐
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            # ---- 第二层全连接层 ----
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            # ---- 第三层全连接层 ----
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                # 初始化weight是正态分布(0,0.01)， bias=0
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)


if __name__ == '__main__':
    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    # tbwriter = SummaryWriter()
    print('TensorboardX summary writer created')

    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)
    print('AlexNet created')

    # create dataset and data loader
    # ImageFolder要求DIR里布局如下，整理成每个样本(image, label)元组形式
    # TRAIN_IMG_DIR/
    #   ├── class_0/       # 标签 0
    #   │   ├── img1.jpg
    #   │   └── img2.jpg
    #   └── class_1/       # 标签 1
    #       ├── img3.jpg
    #       └── img4.jpg
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset created')
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True, # 将数据预加载到 GPU 内存（加速 GPU 训练）。
        num_workers=8,
        drop_last=True, # 丢弃最后一个不完全batchsize的批次
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    # create optimizer
    # the one that WORKS
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY)
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs  每30轮降1/10
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    # start training!!
    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # calculate the loss
            output = alexnet(imgs) # (batch, numclasses)

            # 交叉熵 -log(softmax()) ，  softmax值是0~1，log后为负的，加负号变成正的。 越小越好
            # log_probs = F.log_softmax(output, dim=1) # 对output做softmax(此处log是为数值稳定)
            # loss = F.nll_loss(log_probs, classes) # 对classes中的对应索引处的softmax后的值取-log，得到loss
            loss = F.cross_entropy(output, classes)

            # update the parameters
            # 在PyTorch中，梯度是累积的。每次调用 loss.backward() 时，计算出的梯度会累加到原有的梯度上（而不是替换）。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            # 记录, 图像太少了就没到
            if total_steps % 10 == 0:
                with torch.no_grad():
                    # 最高概率值，索引号
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)
                    tbwriter.flush()
            # print out gradient values and parameter average values
            # 利于查看某层的梯度是否爆炸/消失； 
            # 监控权重变化，查看模型的收敛
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                    parameter.grad.cpu().numpy(), total_steps)
                            tbwriter.flush()
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                    parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
                            tbwriter.flush()
            total_steps += 1

        # save checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
        torch.save(state, checkpoint_path)