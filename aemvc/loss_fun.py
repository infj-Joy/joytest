import torch
import torch.nn as nn

import pre_data
import numpy as np

from scipy.stats import pearsonr


#计算两个表示 H1 和 H2 之间的余弦相似性损失
def corre_loss(H1, H2):
    c = torch.cosine_similarity(H1, H2, dim=1)
    nc = -torch.norm(c, p=1)
    return nc
    # c = pearsonr(H1, H2)
    #
    # return -torch.norm(c, 1)

#实现图正则化损失，保持潜在表示 z 的局部结构
def graph_reg(L1, z, D, lamb):
    k = torch.matmul(z.t(), D)
    st = torch.matmul(k, z)
    m_loss = nn.MSELoss()
    loss = lamb[0] * torch.trace(torch.matmul(torch.matmul(z.t(), L1), z)) +\
           lamb[1] * m_loss(st, torch.eye(st.shape[0]).cuda())
    return loss

#计算拉普拉斯损失，增强表示 z 与核矩阵 K 的关联
def lap_loss(z, K):
    Z = torch.matmul(z, z.t())
    loss = 1/2 * torch.trace(torch.matmul(K.t(), Z))
    return loss

#计算交叉熵损失，用于分类任务
def ce_loss(logits, gt):
    loss = nn.CrossEntropyLoss()
    ce = loss(logits, gt)
    return ce

#计算 HSIC 损失，衡量表示 z 和核矩阵 k2 的相关性
def hsic(z, k2):
    # k1 = pre_data.similarity_cos(z.data.numpy(), z.data.numpy())
    # k1 = pre_data.get_k(z, 'torch')
    k1 = torch.matmul(z, z.t())
    # # k1 = t orch.from_numpy(pre_data.scaler(torch.matmul(z, z.t()).data.numpy()))
    n = k1.size(0)
    e = torch.ones(n, n)
    H = (torch.eye(n) - (1/n) * e).cuda()
    hsic = -((n-1)**(-2))*(torch.trace(torch.matmul(torch.matmul(torch.matmul(k1, H), k2), H)))
    # hsic = torch.log(hsic)
    # k12 = torch.matmul(k1, k2)
    # H = torch.trace(k12)/n**2 + torch.mean(k1)*torch.mean(k2) - 2*torch.mean(k12)/n
    # hsic = H*n**2/(n-1)**2
    # hsic = -torch.log(hsic)

    return hsic

