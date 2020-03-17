'''
@Autor: xujiahuan
@Date: 2020-03-16 17:32:11
@LastEditors: xujiahuan
@LastEditTime: 2020-03-16 17:35:18
'''
import torch
import torchvision
import numpy as np
import sys
sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

print(torch.__version__)
print(torchvision.__version__)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
