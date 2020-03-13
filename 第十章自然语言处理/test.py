'''
@Autor: xujiahuan
@Date: 2020-03-05 23:01:11
@LastEditors: xujiahuan
@LastEditTime: 2020-03-13 23:54:44
'''
import torch
import os
import collections
import random
import math
import sys
import torch.utils.data as Data
from torch import nn
import time

print(torch.__version__)

assert 'ptb.train.txt' in os.listdir("../data/ptb")

with open('../data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st是sentence的缩写
    raw_dataset = [st.split() for st in lines]

print('# sentences: %d' % len(raw_dataset))

# 建立词库的索引
counter = collections.Counter([tk for st in raw_dataset for tk in st])
# key为word,value为出现次数
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

# 存放了所有的单词
idx_to_token = [tk for tk, _ in counter.items()]
# 一个dict，保存了每个单词的id
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
# 每句话的单词的idx表示
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
print('# tokens: %d' % num_tokens)


# 二次采样
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)


subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
print('# tokens: %d' % sum([len(st) for st in subsampled_dataset]))


def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))


# 提取中心词和背景词
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
print("all centers长度:", len(all_contexts))


# 负采样
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)


# 读取数据
def batchify(data):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list,
    list中的每个元素都是__getitem__得到的结果"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index],
                self.negatives[index])

    def __len__(self):
        return len(self.centers)


batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4

dataset = MyDataset(all_centers,
                    all_contexts,
                    all_negatives)
# DataLoader的用法
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                            collate_fn=batchify,
                            num_workers=num_workers)
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks',
                           'labels'], batch):
        print(name, 'shape:', data.shape)
    break

# 跳字模型
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(embed.weight)


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


# 损失函数
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):  # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs,
                                                             targets,
                                                             reduction="none",
                                                             weight=mask)
        return res.mean(dim=1)


loss = SigmoidBinaryCrossEntropyLoss()
pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 标签变量label中的1和0分别代表背景词和噪声词
label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
# 掩码变量(是否是填充词)
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
signal_loss = loss(pred, label, mask)


def sigmd(x):
    return - math.log(1 / (1 + math.exp(-x)))


# 初始化模型参数
embed_size = 100
net = nn.Sequential(
    # 第一层是作为中心词的向量
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
    # 第二层是作为背景词的向量
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)
)


# 定义训练函数
def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for
                                                     d in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            # 使用掩码变量mask来避免填充项对损失函数计算的影响
            # 一个batch的平均loss
            loss_ = (loss(pred.view(label.shape), label, mask) *
                     mask.shape[1] / mask.float().sum(dim=1)).mean()
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            l_sum += loss_.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))


train(net, 0.01, 10)


# 应用词嵌入模型
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) *
                                torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))


get_similar_tokens('chip', 3, net[0])
