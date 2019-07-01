from torch import nn
import torch as t
from torch.nn import functional as F
'''
 残差块
 in_channels, out_channels：残差块的输入、输出通道数
 对第一层，in out channel都是64，其他层则不同
 对每一层，如果in out channel不同， stride是1，其他层则为2
'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels))
        else:
        # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet34(nn.Module):
    #ResNet 34
    def __init__(self,block):
        super(ResNet34,self).__init__()
        # 初始卷积层核池化层
        # cifar 32*32*3
        self.first = nn.Sequential(
            # 卷基层1：7*7kernel，2stride，3padding，outmap：32-7+2*3 / 2 + 1，16*16
            nn.Conv2d(3,64,7,2,3,bias=False), #32*32*3->16*16*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 最大池化，3*3kernel，1stride（32的原始输入图片较小，不再缩小尺寸），1padding，
            # outmap：16-3+2*1 / 1 + 1，16*16
            nn.MaxPool2d(3,1,1), #16*16*64->16*16*64
        )
        # 第一层，通道数不变
        self.layer1 = self.make_layer(block, 64, 64, 3, 1)

        # 第2、3、4层，通道数*2，图片尺寸/2
        self.layer2 = self.make_layer(block, 64, 128, 4, 2)  # 输出8*8
        self.layer3 = self.make_layer(block, 128, 256, 6, 2)  # 输出4*4
        self.layer4 = self.make_layer(block, 256, 512, 3, 2)  # 输出2*2

        self.avg_pool = nn.AvgPool2d(2)  # 输出512*1
        self.fc = nn.Linear(512, 10)

    def make_layer(self,block,in_channels,out_channels,block_num,stride):
        layers = []
        # 每一层的第一个block，通道数可能不同
        layers.append(block(in_channels,out_channels,stride))

        # 每一层的其他block，通道数不变，图片尺寸不变
        for i in range(block_num-1):
            layers.append(block(out_channels,out_channels,1))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

def main():
    net = ResNet34(ResBlock)
    print (net)
if __name__ == '__main__':
    main()