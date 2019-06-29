from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torchvision as tv
import torch as t
import torchvision.models as models
import math

#https://blog.csdn.net/zhanghao3389/article/details/85038252
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def getData():
    transform = transforms.Compose([transforms.Scale(224),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 训练集
    trainset = tv.datasets.CIFAR10(root='./data/',
                                   train=True, download=True, transform=transform)
    trainloader = t.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    # 测试集
    testset = tv.datasets.CIFAR10(root='./data/',
                                  train=False, download=True, transform=transform)
    testloader = t.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    return trainloader,testloader

def runvgg(vgg):
    trainloader, testloader = getData()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #print ("i:",i)
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            #print(inputs.size(), labels)
            #inputs = inputs.permute(0,3,2,1)
            #print (inputs.size(),labels)
            optimizer.zero_grad()
            # forward + backword
            outputs = vgg(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5 == 4:
                print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

class VGG(nn.Module):
    def __init__(self,features,num_classes=1000,init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,10))

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        #如下操作进行初始化方法是 PyTorch 作者所推崇的：
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v=='M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
def vgg11(**kwargs):
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model

def main():
    model_vgg11 = vgg11()
    runvgg(model_vgg11)
    #print (model_vgg11)
    #print (t.cuda.device_count())
if __name__ == '__main__':
    main()