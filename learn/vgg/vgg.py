from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torchvision as tv
import torch as t

#https://blog.csdn.net/zhanghao3389/article/details/85038252
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        #input is 224*224*3
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),# 224*224*3-> 224*224*64
            nn.ReLU(),
            nn.MaxPool2d(2,2),#224*224*64 -> 112*112*64

            nn.Conv2d(64,128,3,padding=1),#112*112*64->112*112*128
            nn.ReLU(),
            nn.MaxPool2d(2,2), #56*56*128

            nn.Conv2d(128,256, 3, padding=1), #56*56*128->56*56*256
            nn.ReLU(),
            nn.Conv2d(256,256, 3, padding=1),#56*56*256->56*56*256
            nn.ReLU(),
            nn.MaxPool2d(2, 2),# 28*28*256

            nn.Conv2d(256, 512, 3, padding=1),  # 28*28*256->28*28*512
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),  # 28*28*512->28*28*512
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14*14*512

            nn.Conv2d(512, 512, 3, padding=1),  # 14*14*512->14*14*512
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),  # 14*14*512->14*14*512
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7*7*512

        )

        self.classifier = nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,10)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x


def vgg():
    vgg = VGG()
    #print (vgg)
    transform = transforms.Compose([transforms.Scale(224),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 训练集
    trainset = tv.datasets.CIFAR10(root='./data/',
                                   train=True, download=True, transform=transform)

    trainloader = t.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # 测试集
    testset = tv.datasets.CIFAR10(root='./data/',
                                  train=False, download=True, transform=transform)

    testloader = t.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

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


def main():
    vgg()
    #print (t.cuda.device_count())
if __name__ == '__main__':
    main()