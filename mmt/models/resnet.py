from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import pdb
from .layers import (
    SpatialAttention2d,
    WeightedSum2d)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, is_select=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.is_select = is_select
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0  #   false
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:   #  false
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:                    # 进入这里
                # Change the num_features to CNN output channels
                self.num_features = out_planes  # out_planes = 2048  num_features  重新被赋值 2048
                self.num_features_delg = 512
                self.feat_bn = nn.BatchNorm1d(self.num_features_delg)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features_delg, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        ## wangzy  add  attention
        self.attention = SpatialAttention2d(in_c=self.num_features, act_fn='relu')
        self.weightSum = WeightedSum2d()

        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False):
        x = self.base(x)    # b x c x H x w    C = 2048  即：32  2048  16  8
        # 1*1 conv 512
        original_fea = x
        # x = self.gap(x)
        # x = x.view(x.size(0), -1)
        '''wangzy add attention'''

        x, att_score = self.attention(x) # 32 1 16 8 比如说取前64个
        # x  torch.Size([32, 512, 16, 8])   att_score  torch.Size([32, 1, 16, 8])
        # print(att_score)
        # x = self.weightSum([x,att_score])#回乘att_score分数
        x = self.gap(x)  # 32*512*1*1
        # print('------------------------------------------------------------')
        # print(x)
        x = x.view(-1, x.size()[1])   # 32 512
        features = x
        # print("features:",features.shape)
        # pdb.set_trace()

        if self.cut_at_pooling:     # False
            return features
        if self.has_embedding:      # false
            bn_x = self.feat_bn(self.feat(features))
        else:                       # 进入这里
            bn_x = self.feat_bn(features)

        # print("training:", self.training)  ### 不确定！
        if self.training is False:  ##  分情况  pretrain的时候 应该是 true   target finetune  确定是 false
            prob = self.classifier(bn_x)
            bn_x = F.normalize(bn_x)
            return bn_x, prob, original_fea, att_score  ### !!!! finetune 的时候从这里 return
            # return bn_x, self.feat_bn(original_fea), att_score   ### !!!! finetune 的时候从这里 return

        if self.norm:               # False
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:        # False
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:    # True
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn:          # False
            return bn_x, prob

        return features, prob, original_fea, att_score
        #att_score (16,1,16,8)
        #original_fea(16,2048,16,8)
        #prob (16,12936)
        #features (16,2048)


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.maxpool.state_dict())
        self.base[3].load_state_dict(resnet.layer1.state_dict())
        self.base[4].load_state_dict(resnet.layer2.state_dict())
        self.base[5].load_state_dict(resnet.layer3.state_dict())
        self.base[6].load_state_dict(resnet.layer4.state_dict())

def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
