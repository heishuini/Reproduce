'''
https://blog.csdn.net/qq_73462282/article/details/132191515
todo.
'''

import torch.nn as nn
import torchvision.models as models
import torch


if __name__ == "__main__":
    # 用官方预训练权重
    model = models.resnet18(pretrained=True)

    #####  迁移学习
    # 卷积层不进行更改
    for p in model.parameters():
        p.requires_grad = False

    # 修改模型的全连接层
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f,4)

    # 模型转移到pytorch里
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.fc.parameters(),lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    
    
    dataloader = None
    
    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    ###### 
    
