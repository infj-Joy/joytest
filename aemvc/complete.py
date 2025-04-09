import torch

import pre_data
import numpy as np


#定义complete函数：Kycc是核矩阵，x1是特征矩阵（完整）
def complete(Kycc, x1):
    #计算相似度矩阵和核矩阵
    w = pre_data.similarity_cos(x1.cpu().data.numpy(), x1.cpu().data.numpy())
    # K1 = pre_data.get_k(x1.data.numpy(), 'torch')
    K1 = torch.matmul(x1, x1.t())
    #计算中心矩阵和 HSIC 矩阵
    n = K1.size(0)
    e = torch.ones(n, n)
    H = (torch.eye(n) - (1 / n) * e).cuda()

    #HSIC约束性计算
    sigma = -(0)*((n-1)**(-2))*torch.matmul(torch.matmul(H, K1), H) # HSIC
    # sigma = torch.from_numpy(pre_data.scaler(sigma.data.numpy()))

    #计算图拉普拉斯矩阵
    L = pre_data.getNormLaplacian(w).numpy()
    # L = pre_data.scaler(L)
    L = (torch.from_numpy(L)).cuda()

    #划分矩阵并计算子矩阵
    c = Kycc.size(0)
    m = w.shape[0]-c
    L_cm = 2*L[0:c, c:(c + m)] + sigma[0:c, c:(c + m)]
    L_mm = 2*L[c:(c + m), c:(c + m)] + sigma[c:(c + m), c:(c + m)]
    L_mm_1 = torch.inverse(L_mm)

    #矩阵逆和变换
    Kycm = (torch.matmul(torch.matmul(-Kycc, L_cm), L_mm_1)).cpu().data.numpy()
    Kymm = (torch.matmul(torch.matmul(torch.matmul(torch.matmul(L_mm_1, L_cm.t()), Kycc), L_cm), L_mm_1)).cpu().data.numpy()
    Kymc = (torch.matmul(torch.matmul(-L_mm_1, L_cm.t()), Kycc)).cpu().data.numpy()
    #使用预处理模块对矩阵进行标准化
    Kycm, Kymm, Kymc = torch.from_numpy(pre_data.scaler(Kycm)), \
                       torch.from_numpy(pre_data.scaler(Kymm)), \
                       torch.from_numpy(pre_data.scaler(Kymc))
    
    #拼接子矩阵，构建最终的核矩阵
    Ky0 = torch.cat((Kycc.cpu(), Kycm), dim=1)
    Ky1 = torch.cat((Kymc, Kymm), dim=1)
    Ky = (torch.cat((Ky0, Ky1), dim=0)).numpy()
    Ky = ((Ky + Ky.T) / 2).astype(np.float32)
    np.fill_diagonal(Ky, 1.0)

    Ky = (torch.from_numpy(Ky)).cuda()
    return Ky