import torch.nn as nn

# 样本读取线程数
WORKERS = 4

# 网络参赛保存文件名
PARAS_FN = 'cifar_resnet_params.pkl'

# minist数据存放位置
ROOT = './data'

# 目标函数
loss_func = nn.CrossEntropyLoss()

# 记录准确率，显示曲线
global_train_acc = []
global_test_acc = []

def main():
    print ("success")
if __name__ == '__main__':
    main()