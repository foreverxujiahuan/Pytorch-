import torch
import numpy as np
torch.manual_seed(0)

# 获取torch版本
version = torch.__version__

# 创建Tensor
x = torch.empty(5, 3)

# 创建long型的全0的Tensor
long_tensor = torch.zeros(5, 3, dtype=torch.long)

# 根据数据创建tensor
data_x = torch.tensor([5.5, 3])

# 通过shape来获取tensor的形状
shape = x.shape

# 加法
x = torch.rand(5, 3)
y = torch.rand(5, 3)
res = x+y

# inplace加法
y.add_(x)

# 改变形状
y = x.view(15)
y = x.view(3, 5)

# torch和numpy的互换
array = [1, 2, 3, 4, 5]
numpy_array = np.array(array)
torch_array = torch.tensor(numpy_array)
torch_array2 = torch.from_numpy(numpy_array)

# 使用GPU
flag = torch.cuda.is_available()
gpu_array = torch_array.cuda()
