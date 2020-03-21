'''
@Autor: xujiahuan
@Date: 2020-03-17 23:28:11
@LastEditors: xujiahuan
@LastEditTime: 2020-03-18 00:15:18
'''
import torch
from torch import nn
from torch.nn import init
import numpy as np
from collections import OrderedDict
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)

# 定义batch_size
batch_size = 256
# 定义训练数据和测试数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 输入维度,28*28
num_inputs = 784
# 输出维度,label可能有十种
num_outputs = 10


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
          ('flatten', FlattenLayer()),
          ('linear', nn.Linear(num_inputs, num_outputs))])
        )

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = nn.CrossEntropyLoss()
# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# 定义训练次数
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
