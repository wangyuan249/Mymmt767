from __future__ import print_function, absolute_import

import math
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import heapq
import pdb

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .utils.meters import AverageMeter

def dotProduct(v1, v2):
    v1 = np.mat(v1)
    v2 = np.mat(v2)
    z = v1 * v2.T
    return z

class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out, _, _ = self.model(s_inputs)
            # target samples: only forward
            t_features, _, _, _  = self.model(t_inputs)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

class ClusterBaseTrainer(object):
    def __init__(self, model, num_cluster=500):
        super(ClusterBaseTrainer, self).__init__()
        self.model = model
        self.num_cluster = num_cluster

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()

    def train(self, epoch, data_loader_target, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            # forward
            f_out_t, p_out_t, focal_tar = self.model(inputs)
            p_out_t = p_out_t[:,:self.num_cluster]

            # sim_target = torch.sum(p_out_t * focal_tar, 1) / torch.sqrt(
            #     torch.sum(torch.pow(p_out_t, 2), 1)) / torch.sqrt(torch.sum(torch.pow(focal_tar, 2), 1))

            loss_ce = self.criterion_ce(p_out_t, targets)
            loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)
            loss = loss_ce + loss_tri + similarity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec, = accuracy(p_out_t.data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tri {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset, TensorDataset

class MMTTrainer(object):
    def __init__(self, model_1, model_2,
                       model_1_ema, model_2_ema, num_cluster=500, alpha=0.999):
        super(MMTTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
    def select_byentropy(self, epoch, data_loader_target,
                         len_dataset, batch_size, print_freq = 20,):
        global sum_dataset
        self.select_iters = len_dataset / batch_size + 1
        self.logsoftmax = torch.nn.LogSoftmax(dim=0).cuda()
        self.softmax = torch.nn.Softmax(dim=0).cuda()
        self.model_1.train()
        self.H_threshold = 2
        fpath_list = []
        unlabel_fpath_list = []
        label_pathonly_list = []
        for i in range(int(self.select_iters)):
            target_inputs = data_loader_target.next()

            # process inputs
            inputs_1, inputs_2, targets, fpath = self._parse_data(target_inputs)
            # inputs_1:  torch.Size([32, 3, 256, 128])
            # targets:  torch.Size([32])
            # print("fpath: ", fpath.shape)

            # forward
            f_out_t1, p_out_t1, ori_fea_t1, focal_tar_t1= self.model_1(inputs_1)
            p_out_t1 = p_out_t1[:,:self.num_cluster]   # b * 聚类个数 比如 806

            # select
            index_list = []
            otherindex_list = []
            for j in range(0, p_out_t1.size(0)):
                mytensor = p_out_t1[j,:]
                # print("mytensor: ", mytensor.shape)
                log_probs = self.logsoftmax(mytensor)  # 例 [54]

                softmax_prob = self.softmax(mytensor)
                # print("mytensor", mytensor)
                # print("log_probs",log_probs)
                # print("softmax_prob", softmax_prob)
                H = -torch.sum(softmax_prob * log_probs, dim=0)  # 熵 分布越平均，熵越大
                # print("H: ", H)
                if H < self.H_threshold:
                    index_list.append(j)
                else :
                    otherindex_list.append(j)

            select_len = len(index_list)

            for index in index_list:
                labels = int(targets[index].cpu().detach())
                tuple = (fpath[index], labels, 0)
                # fpath_list.append(fpath[index])
                fpath_list.append(tuple)
                label_pathonly_list.append(fpath[index])

            for index in otherindex_list:
                labels = int(targets[index].cpu().detach())
                tuple = (fpath[index], labels, 0)
                # fpath_list.append(fpath[index])
                unlabel_fpath_list.append(tuple)

            for j in range(len(index_list),batch_size):
                 index_list.append(0)                   ## 补全维度
            # print("index_list",len(index_list))
            # index_list = torch.tensor(index_list).cuda()
            # index_list_tmp = torch.tensor(index_list).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(inputs_1).cuda()
            # new_inputs = torch.gather(inputs_1, dim=0, index=index_list_tmp)
            # new_targets = torch.gather(targets, dim=0, index=index_list)
            # new_inputs = new_inputs[:select_len]
            # new_targets = new_targets[:select_len]
            new_inputs = inputs_1[:select_len]
            new_targets = targets[:select_len]
            # 将数据封装成subDataset
            sub_dataset = TensorDataset(new_inputs, new_targets)
            # 将子数据集做拼接封装成Dataset
            # sum_dataset.datasets.append(sub_dataset)
            if i == 0:
                sum_dataset = sub_dataset
            else:
                sum_dataset = ConcatDataset([sum_dataset, sub_dataset])
            # print("sum_dataset", len(sum_dataset))


        #return sum_dataset, fpath_list, unlabel_fpath_list
        return sum_dataset, fpath_list, label_pathonly_list


    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=20, train_iters=200, cos_soft_weight=0.5,
              loss_2norm_weight = 0.025):
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_cos = [AverageMeter(),AverageMeter()]
        losses_2norm = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        losses_cos_soft = AverageMeter()

        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, targets, fpath = self._parse_data(target_inputs)

            # forward
            f_out_t1, p_out_t1, ori_fea_t1, focal_tar_t1= self.model_1(inputs_1)
            f_out_t2, p_out_t2, ori_fea_t2, focal_tar_t2= self.model_2(inputs_2)
            p_out_t1 = p_out_t1[:,:self.num_cluster]
            p_out_t2 = p_out_t2[:,:self.num_cluster]
            # print('-------------------------1--------------------------')
            # print(f_out_t1.shape)#[64, 2048]
            # print('-------------------------2--------------------------')
            # print(p_out_t1.shape)#[64, 809]
            # print('-------------------------3--------------------------')
            # print(ori_fea_t1.shape)#[64,2048,16,8]
            # print('-------------------------4--------------------------')
            # print(focal_tar_t1.shape)#[64, 1, 16, 8]

            f_out_t1_ema, p_out_t1_ema, ori_fea_t1_ema, focal_tar_t1_ema = self.model_1_ema(inputs_1)
            f_out_t2_ema, p_out_t2_ema, ori_fea_t2_ema, focal_tar_t2_ema = self.model_2_ema(inputs_2)
            p_out_t1_ema = p_out_t1_ema[:,:self.num_cluster]
            p_out_t2_ema = p_out_t2_ema[:,:self.num_cluster]

            #纭В锟?
            #model
            focal_tar_t1 = focal_tar_t1.cpu().detach().numpy()
            focal_tar_t2 = focal_tar_t2.cpu().detach().numpy()

            #att_score  鑼冩暟绾︽潫
            loss_2norm_max = 28
            loss_2norm = abs(2*loss_2norm_max - np.linalg.norm(focal_tar_t1) - np.linalg.norm(focal_tar_t2))*loss_2norm_weight

            #纭В锟?
            # share_att_t1 = (focal_tar_t1 - np.min(focal_tar_t1)) / (1.0 * (np.max(focal_tar_t1) - np.min(focal_tar_t1)))
            # share_att_t2 = (focal_tar_t2 - np.min(focal_tar_t2)) / (1.0 * (np.max(focal_tar_t2) - np.min(focal_tar_t2)))

            # unique_att_t1 = share_att_t1
            # share_att_t1 = np.where(share_att_t1 > np.median(share_att_t1), share_att_t1, 0)#鑾峰彇纭В鑰
            # unique_att_t1 = np.where(unique_att_t1 < np.median(unique_att_t1), unique_att_t1, 1)  # 鑾峰彇纭В锟?-a
            # unique_att_t1 = 1 - unique_att_t1
            #
            # unique_att_t2 = share_att_t2
            # share_att_t2 = np.where(share_att_t2 > np.median(share_att_t2), share_att_t2, 0)#鑾峰彇纭В鑰
            # unique_att_t2 = np.where(unique_att_t2 < np.median(unique_att_t2), unique_att_t2, 1)  # 鑾峰彇纭В锟?-a
            # unique_att_t2 = 1 - unique_att_t2

            #纭В锟?
            share_att_t1 = focal_tar_t1
            share_att_t2 = focal_tar_t2

            #print(focal_tar_t1.shape,ori_fea_t1.shape)
            #鎻愬彇鍓?/3
            unique_att_t1 = share_att_t1
            reshape_att_t1 = share_att_t1.reshape(share_att_t1.shape[0],share_att_t1.shape[1],-1)
            sort_att_t1 = np.argsort(reshape_att_t1)
            one_third_sum_t1 = 0.0
            for i in range(reshape_att_t1.shape[0]):
                one_third_sum_t1 += reshape_att_t1[i][0][sort_att_t1[i][0][int(reshape_att_t1.shape[2] * 1 / 3)]]
            one_third_t1 = one_third_sum_t1/share_att_t1.shape[0]
            unique_att_t1 = np.where(unique_att_t1 < one_third_t1, unique_att_t1, 0)  # 鑾峰彇纭В鑰?-a att鍓?/3閮ㄥ垎 鍩熷叡浜儴鍒?

            unique_att_t2 = share_att_t2
            reshape_att_t2 = share_att_t2.reshape(share_att_t2.shape[0], share_att_t2.shape[1], -1)
            sort_att_t2 = np.argsort(reshape_att_t2)
            one_third_sum_t2 = 0.0
            for i in range(reshape_att_t2.shape[0]):
                one_third_sum_t2 += reshape_att_t2[i][0][sort_att_t2[i][0][int(reshape_att_t2.shape[2] * 1 / 3)]]
            one_third_t2 = one_third_sum_t2/share_att_t2.shape[0]
            unique_att_t2 = np.where(unique_att_t2 < one_third_t2, unique_att_t2, 0)  # 鑾峰彇纭В鑰?-a


            # print(one_third_t1,np.sum(unique_att_t1 <= 0.0), np.sum(unique_att_t2 <= 0.0),np.median(share_att_t1),
            #       np.sum(share_att_t1),np.sum(unique_att_t1))
            # print(one_third_t2,np.sum(unique_att_t2 <= 0.0), np.sum(unique_att_t2 <= 0.0),np.median(share_att_t2),
            #       np.sum(share_att_t2),np.sum(unique_att_t2))

            share_focal_t1 = (ori_fea_t1.cpu().detach()*torch.tensor(share_att_t1))
            unique_focal_t1 = (ori_fea_t1.cpu().detach()*torch.tensor(unique_att_t1))

            share_focal_t2 = (ori_fea_t2.cpu().detach()*torch.tensor(share_att_t2))
            unique_focal_t2 = (ori_fea_t2.cpu().detach()*torch.tensor(unique_att_t2))

            # print("share_focal_t1.shape : ", share_focal_t1.numpy().shape)
            # print("share_focal_t1.__class__: ", share_focal_t1.numpy().__class__)
            # similarity_list = []
            # share_focal = share_focal_t1.numpy()
            # for i in range(0, share_focal.size(2)):
            #     for j in range(0, share_focal.size(3)):
            #         vector1 = share_focal[:, :, i]
            #         vector2 = share_focal[:, :, j]
            #         cosine = dotProduct(vector1, vector2) / \
            #                  math.sqrt(dotProduct(vector1, vector1) * dotProduct(vector2, vector2))
            #         print("cosine: ", cosine)
            #         similarity_list.append(cosine)
            #
            # similarity_list = np.array(similarity_list)

            # np.save('./similarity_list_3.npy', share_focal_t1.numpy())
            # print("similarity_list_2.shape: ", share_focal_t1.numpy().shape)
            # pdb.set_trace()

            similarity_weight = 5
            similarity_t1 = torch.cosine_similarity(share_focal_t1.view(32,-1), unique_focal_t1.view(32,-1), dim=1)
            similarity_t2 = torch.cosine_similarity(share_focal_t2.view(32,-1), unique_focal_t2.view(32,-1), dim=1)
            similarity_t1 = torch.mean(similarity_t1) * similarity_weight
            similarity_t2 = torch.mean(similarity_t2) * similarity_weight

            #loss
            loss_ce_1 = self.criterion_ce(p_out_t1, targets)
            loss_ce_2 = self.criterion_ce(p_out_t2, targets)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)
            loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
                            self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)

            loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
                     (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                     (similarity_t1 + similarity_t2)+ \
                    loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight + \
                   loss_2norm
            # print(loss)
            # loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
            #          (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
            #         loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight + \
            #        loss_2norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)
            prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri[1].update(loss_tri_2.item())
            losses_cos[0].update(similarity_t1.item())
            losses_cos[1].update(similarity_t2.item())
            losses_2norm.update(loss_2norm.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            # losses_cos_soft.update(loss_similarity_soft.item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_cos {:.3f} / {:.3f}\t'
                      'Loss_2norm {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_cos[0].avg, losses_cos[1].avg,
                              losses_2norm.avg,losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg, precisions[1].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids, fpath = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets, fpath
