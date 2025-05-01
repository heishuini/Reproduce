'''
 https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L684
'''

import torch.nn as nn
from typing import Optional, Callable, Union
from torch import Tensor
import torch

def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1, 
            groups: int = 1, 
            dilation:int = 1,
        ) -> nn.Conv2d:
    
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,     # 分组卷积， 普通卷积是输入通道和输出通道全连接(group=1)，此处分组的话可将通道分为多组独立卷积，轻量化
        bias=False,
        dilation=dilation, # 膨胀卷积，为了增加感受野，计算方式：k_eff=k+(k-1)x(d-1) 感受野扩大，
                           # 但是计算的时候是跳过中间像素，仍然计算kxk个点
    )

def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1, 

)-> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )

'''
ResNetBlock: Basic , 主要用于构建ResNet18, ResNet34
    64-d 
     ↓
    3x3,64
    BN, ReLU
    3x3,64
    BN
     ↓
     +x
     ReLU
'''
class BasicBlock(nn.Module):
    expansion: int = 1
    
    def __init__(self, 
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 # 可选参数，可能是nn.Module或None
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 # 调用后返回nn.Module, ...表示任意参数
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        
        # 默认归一化层为BN, 需要输入张量的通道数作为参数
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        
        # 膨胀卷积，同时作为padding填充步
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # 
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes) # 可使用零初始化
        
        # 可选
        self.downsample = downsample
        self.stride = stride
        
    def forawrd(self, x : Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # downsample: 用于通道数翻倍时空间减半，conv1x1+BN
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out

'''
针对更多的通道数，为减小计算量
 256-d
   ↓
 1x1,64
 BN,ReLU
 3x3,64
 BN,ReLU 
 1x1(x4),256
 BN(x4)
   ↓
   +x
   ReLU
'''

class Bottleneck(nn.Module):
    expansion: int = 4
    
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 输入通道数，可变
        width = int(planes * (base_width / 64.0)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width) 
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width) 
        # 默认是4倍，增加通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True) # 对x直接修改
        # downsample: 用于通道数翻倍时空间减半，conv1x1+BN
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x : Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(x) # 通道数扩大4倍
        out = self.bn3(out) # 通道数扩大4倍, 可使用0初始化
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
    
class ResNet(nn.Module):
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]], # type表示类(不是实例),Union表示或关系
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,        
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        
        ###### 是否用空洞卷积(dilated convolution)替代常规的步长(stride)下采样的配置逻辑
        # 需要保持较高分辨率特征图时(如语义分割任务)，可以用空洞卷积替代步长下采样
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        ###### 
        
        self.groups = groups
        self.base_width = width_per_group
        # 输入层: 7x7卷积, h = (h+2x3-7)/2 + 1 = (h-1)/2 + 1 向下取整
        # 如224x224 -> 112x112, 分辨率减半
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # h = (h+2x1-3)/2 + 1 = (h-1)/2 + 1
        # 分辨率减半
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # resnet18(BasicBlock,[2, 2, 2, 2])
        self.layer1 = self._make_layer(block, 64, layers[0])
        # stride=2的目的，因为通道
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        # 全部压成1x1空间大小
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 零初始化残差分支最后一个BN层
        # 核心目的是通过特殊初始化方式，让残差网络（ResNet）在训练初期表现得像恒等映射
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        
    def _make_layer(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # 本层通道数是前面一层的两倍，作为接入时，通过conv1x1实现 (如64->128),+ BN
        # 同时，stride=2，h=(h+0-1)/2 + 1 , 空间分辨率减半
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            
        layers = []
        # 第一个块, stride=2接收(若有,第一层没有stride)
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        
        # expansion
        # BasicBlock: 1
        # BottleNeck: 4
        self.inplanes = planes * block.expansion
        
        # 中间块
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
            
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x) # 7x7卷积, 64channel
        x = self.bn1(x) 
        x = self.relu(x)
        
        x = self.maxpool(x) # 空间减半
        
        x = self.layer1(x) # 64
        x = self.layer2(x) # 128
        x = self.layer3(x) # 256
        x = self.layer4(x) # 512
        
        x = self.avgpool(x) # 空间压成1x1
        x = torch.flatten(x, 1) # 从第一维开始，后续维度全部合并为一个维度 (B,C,H,W) -> (B,C*H*W)
        
        x = self.fc(x) # 全连接输出层 512*expansion -> num_classes
        
        return x
        
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock,[3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])