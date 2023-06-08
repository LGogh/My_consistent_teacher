import torch
import torch.nn as nn
from mmdet.core import multi_apply
from mmcv.cnn import MODELS

# @MODELS.register_module()
class AugWeight(nn.Module):
    def __init__(self, transforms_len=9):
        super(AugWeight, self).__init__()
        # data_group[] img 的类型是 tensor
        # 图片大小不一样
        # 增强方式是否改变 img.shape ? -> 同一 batch 内 shape 应尽量相同
        # (c, h, w) -> (c, a, b) -> (d)
        # pooling / attention -> 不同大小输入得到相同大小输出

        self.pool_size = 9
        # conv
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        # AdaptiveAvgPool2d() 会得到 inf 
        self.pooling = nn.AdaptiveMaxPool2d(self.pool_size)
        self.fc1 = nn.Linear(self.pool_size * self.pool_size * 3, 64)
        self.fc_trans = nn.Linear(64, transforms_len)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()


    # def forward(self, imgs):       
    #     return multi_apply(self.forward_single, imgs)
    
    # def forward_single(self, img):
    def forward(self, img):
        if torch.isnan(img).any() or torch.isinf(img).any():
            img = torch.where(torch.isnan(img), torch.full_like(img, 0), img)
            img = torch.where(torch.isinf(img), torch.full_like(img, 1), img)

        feat = self.conv1(img)
        feat = self.conv2(feat)
        # feat = self.norm(feat)
        feat = self.relu(feat)
        # (b, c, h, w) -> (b, c, 9, 9)
        feat = self.pooling(img)
        # (b, c*9*9)
        feat = feat.flatten(start_dim=1)
        # (b, 64)
        feat = self.fc1(feat)
        feat = self.relu(feat)
        # (b, 9)
        feat_trans = self.fc_trans(feat)
        # (b, 9)
        trans_prob = self.sigmoid(feat_trans)

        return trans_prob

# class AugWeight(nn.Module):
#     def __init__(self, transforms_len=9, geometric_len=3):
#         super(AugWeight, self).__init__()
#         # data_group[] img 的类型是 tensor
#         # 图片大小不一样
#         # 增强方式是否改变 img.shape ? -> 同一 batch 内 shape 应尽量相同
#         # (c, h, w) -> (c, a, b) -> (d)
#         # pooling / attention -> 不同大小输入得到相同大小输出

#         self.pool_size = 9
#         # conv
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
#         # AdaptiveAvgPool2d() 会得到 inf 
#         self.pooling = nn.AdaptiveMaxPool2d(self.pool_size)
#         self.fc1 = nn.Linear(self.pool_size * self.pool_size * 3, 64)
#         self.fc_trans = nn.Linear(64, transforms_len)
#         self.fc_geome = nn.Linear(64, geometric_len)
#         self.sigmoid = nn.Sigmoid()
#         # self.softmax = nn.Softmax(dim=1)
#         self.relu = nn.ReLU()

#     def forward(self, img):       
#         if torch.isnan(img).any() or torch.isinf(img).any():
#             img = torch.where(torch.isnan(img), torch.full_like(img, 0), img)
#             img = torch.where(torch.isinf(img), torch.full_like(img, 1), img)

#         feat = self.conv1(img)
#         feat = self.conv2(feat)
#         # feat = self.norm(feat)
#         feat = self.relu(feat)
#         # (b, c, h, w) -> (b, c, 9, 9)
#         feat = self.pooling(img)
#         # (b, c*9*9)
#         feat = feat.flatten(start_dim=1)
#         # (b, 64)
#         feat = self.fc1(feat)
#         feat = self.relu(feat)
#         # (b, 9)
#         feat_trans = self.fc_trans(feat)
#         # (b, 9)
#         trans_prob = self.sigmoid(feat_trans)

#         # return trans_prob
#         #(b, 3)
#         feat_geome = self.fc_geome(feat)
#         geome_prob = self.sigmoid(feat_geome)

#         return trans_prob, geome_prob