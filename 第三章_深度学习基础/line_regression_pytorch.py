
import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim
torch.manual_seed(1)

torch.set_default_tensor_type('torch.FloatTensor')

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)),
                        dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float)

# 读取数据
batch_size = 10

# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)

# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)


# 定义模型
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 初始化参数模型
init.normal_(net[0].weight, mean=0.0, std=0.01)
# 也可以直接修改bias的data: net[0].bias.data.fill_(0)
init.constant_(net[0].bias, val=0.0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)

# 训练模型
if __name__ == '__main__': 
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            loss_ = loss(output, y.view(-1, 1))
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
            loss_.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, loss_.item()))

    dense = net[0]
    print(true_w, dense.weight.data)
    print(true_b, dense.bias.data)
