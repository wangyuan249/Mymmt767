import numpy as np
import torch

focal_tar = torch.rand(64, 1, 5, 5)
ori_fea = torch.rand(64, 100, 5, 5)
focal_tar = focal_tar/2.0
focal_tar = focal_tar.numpy()
# print(focal_tar)
# print(np.max(focal_tar))
# for xx in focal_tar:
#     print(xx)
share_att = (focal_tar - np.min(focal_tar)) / (1.0 * (np.max(focal_tar) - np.min(focal_tar)))
# share_att = np.round(np.round((xx - np.min(focal_tar)) / (1.0 * (np.max(focal_tar) - np.min(focal_tar))), 4) for xx in focal_tar)
# print(share_att)
# print('--------------------------')
unique_att = 1 - share_att
# print(unique_att)
# print(torch.from_numpy(share_att))
share_focal = ori_fea * share_att
unique_focal = ori_fea * unique_att
# print(unique_focal.shape)
share_focal_v = share_focal.view(-1)
unique_focal_v = unique_focal.view(-1)
# print(share_focal.shape)
# print(share_focal)
similarity = torch.cosine_similarity(share_focal, unique_focal, dim=0)
similarity_v = torch.cosine_similarity(share_focal_v, unique_focal_v, dim=0)
similarity = similarity.numpy()
similarity_mean = np.mean(similarity)
print(similarity_mean,similarity_v)
