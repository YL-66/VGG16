import copy
import time
import torch
import torch.utils.data as Data
from torch import optim, nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model import VGG16
import pandas as pd

def train_val_data_process():
    train_data = FashionMNIST(root='./data', train=True, transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                              download=True)
    train_data, val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])

    train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=32, shuffle=True, num_workers=2)

    return train_loader, val_loader

train_loader, val_loader = train_val_data_process()

def train_model_process(model,train_loader,val_loader,epochs):
    # 设定训练所用到的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 使用Adam优化器，学习率为0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 模型放入当前设备
    model = model.to(device)
    # 复制当前的模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高的准确度
    best_acc = 0.0
    # 训练集损失列表
    train_losses_all = []
    # 验证机损失列表
    val_losses_all = []
    # 训练集精确度列表
    train_acc_all = []
    # 验证机精度列表
    val_acc_all = []

    since = time.time()

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs-1))
        print('-' * 10)

        # 初始化参数
        # 计算损失函数
        train_loss = 0.0
        # 训练集精确度
        train_corrects = 0.0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0.0
        # 训练集样本数量
        train_num = 0
        # 测试集样本数量
        val_num = 0

        # 对每一个mini_batch训练和计算
        for step, (b_x, b_y) in enumerate(train_loader):
            # 数据也要放到设备里面
            # 特征放入训练设备中
            b_x = b_x.to(device)
            # 标签放入训练设备中
            b_y = b_y.to(device)

            # 设置模型为训练模式
            model.train()

            # 前向传播，输入为一个batch，输出为一个batch中的对应预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output,dim=1)

            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)
            # 将梯度初始化为零
            optimizer.zero_grad()

            # 反向传播
            loss.backward()
            # 根据反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则精确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_loader):
            # 数据也要放到设备里面、
            # 特征放入验证设备中
            b_x = b_x.to(device)
            # 标签放入验证设备中
            b_y = b_y.to(device)
            # 将模型设为评估模式
            model.eval()
            # 前向传播，输入一个batch，输出为一个batch中的对应预测
            output = model(b_x)

            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)

            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则精确度val_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)


        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_losses_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)
        # 计算并保存验证集的loss值
        val_losses_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / train_num)

        # 打印出每一批次的损失值和精确率
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_losses_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_losses_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的权重
        if val_acc_all[-1] > best_acc:
            # 保存当前的最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前的最高准确度
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print(f'训练和验证耗费的时间{time_use//60:.2f}m{time_use%60}s')

    # 选择最优参数，保存最优参数的模型
    # 加载最高准确率下的模型参数
    torch.save(best_model_wts, 'best_model.pth')
    # torch.save(model.load_state_dict(best_model_wts), 'best_model.pth')

    train_process = pd.DataFrame(data={'epoch': range(epochs),
                                       'train_loss_all': train_losses_all,
                                       'train_acc_all': train_acc_all,
                                       'val_loss_all': val_losses_all,
                                       'val_acc_all': val_acc_all})
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'],train_process.train_loss_all,'ro-',label='train loss')
    plt.plot(train_process['epoch'], train_process.val_loss_all, 'bs-', label='val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'ro-', label='train acc')
    plt.plot(train_process['epoch'], train_process.val_acc_all, 'bs-', label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig("acc_loss.png", dpi=150)
    plt.close()  # 关闭画布，释放内存，不要plt.show()


if __name__ == '__main__':
    # 将模型实例化
    VGG16 = VGG16()
    train_loader, val_loader = train_val_data_process()
    train_process = train_model_process(VGG16 , train_loader, val_loader, 20)
    matplot_acc_loss(train_process)






