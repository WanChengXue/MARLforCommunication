import torch
import torch.nn as nn
import torchvision
def conv3x3(in_Planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_Planes, out_planes, kernel_size=3, stride=stride, padding=dilation, \
                    groups = groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 强化学习不需要normalization操作,因此
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ELU()
        self.activate_function = nn.Tanh()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.activate_function(out)
        
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample is not None:
            # 这个意思是说,如果传入的数据和传出来的数据维度不一样,需要进行一次额外的卷积操作
            identity = self.downsample(x)
        out += identity
        out = self.activate_function(out)
        return out

class resnet_34(nn.Module):
    def __init__(self):
        super(resnet_34, self).__init__()
        self.layers = [2,2,2,2]
        self.inplanes = 64
        # 构建第一个卷积层
        self.conv_1 = nn.Conv2d(3, 64, 7, 2, padding=3, bias=False)
        # 这个nn.ReLU()表示直接改变输入的数据
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, self.layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, self.layers[3], stride=2)
        self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, 1000)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            # 这个地方是说,如果stride不是1, 或者说输入的channel数目不是64,都需要一个额外的layer,使得维度一样进行加减
            downsample = nn.Sequential(conv1x1(self.inplanes, planes*block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes *block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # 最前面的几次计算,卷积,relu激活,maxpool进行池化
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 通过4个layer
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
        

    def forward(self, x):
        # 首先对传入的数据进行padding 操作
        return self._forward_impl(x)

