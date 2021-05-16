from __future__ import print_function, absolute_import
import time
import torch.nn as nn
import torch
import numpy as np

import heapq

a = torch.rand(32, 2048, 16, 8)
b = a.view(-1)
c = a.numpy()
# a = a.numpy()
# b = b.numpy()
print(torch.norm(a))
print(torch.norm(b))
print(np.linalg.norm(c))
R = 10
ring_weight = 1
sum_t1 = 0.0
for ori_fea_t1_s in a:
    sum_t1 += (torch.norm(ori_fea_t1_s, p=2) - R).pow(2)
loss_ring_t1 = sum_t1 * ring_weight / a.shape[0] / 2

print(sum_t1,loss_ring_t1)
# gap = nn.AdaptiveAvgPool2d(1)
#
# # print(a)
# share_att_t1 = (a - np.min(a)) / (1.0 * (np.max(a) - np.min(a)))
# unique_att_t1 = share_att_t1
# share_att_t1_mid = np.median(share_att_t1)
# unique_att_t1_mid = np.median(unique_att_t1)
# share_att_t1 = np.where(share_att_t1 > np.median(share_att_t1), share_att_t1, 0)  # 获取硬解耦a
# unique_att_t1 = np.where(unique_att_t1 < np.median(unique_att_t1), unique_att_t1, 1)  # 获取硬解耦1-a
# unique_att_t1 = 1 - unique_att_t1
#
# share_att_t1 = torch.from_numpy(share_att_t1)
# share_att_t1_gap = gap(share_att_t1).view(share_att_t1.shape[0],-1)
#
# unique_att_t1 = torch.from_numpy(unique_att_t1)
# unique_att_t1_gap = gap(unique_att_t1).view(unique_att_t1.shape[0],-1)
#
# share_att_t1 = share_att_t1.view(share_att_t1.shape[0],share_att_t1.shape[1],-1)
# unique_att_t1 = unique_att_t1.view(unique_att_t1.shape[0],unique_att_t1.shape[1],-1)
# # print(share_att_t1.shape)
# #
# # print(share_att_t1_gap)
# # print(unique_att_t1_gap)
# print(share_att_t1.shape)
# print(share_att_t1_gap.shape)
# similarity_t1 = torch.cosine_similarity(share_att_t1, unique_att_t1, dim=2)
# similarity_t1_gap = torch.cosine_similarity(share_att_t1_gap, unique_att_t1_gap, dim=1)
# # similarity_t1_all = torch.cosine_similarity(share_att_t1.view(-1), unique_att_t1.view(-1), dim=0)
# print(similarity_t1.shape,torch.mean(similarity_t1))
# print(similarity_t1_gap.shape,torch.mean(similarity_t1_gap))
# # print(similarity_t1_all.shape,torch.mean(similarity_t1_all))

# #找最小1/3值
# a = torch.rand(5,5)
# a = a.numpy()
#
# # print(a)
# # top_k_idx=a.argsort()
# # print(top_k_idx)
# print(heapq.nlargest(3, a))