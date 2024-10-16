# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
import numpy as np
from detectron2.structures import ImageList
from detectron2.structures import Instances
import math
import pycocotools.mask as mask_utils
from ..backbone import build_backbone
from ..backbone import build_bottom_up_fuse
from ..relation_head import build_relation_head
from ..backbone.panet_fpn import PAnetFPN
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
import torch.nn.functional as F

class build_Salient_predict(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.conv2d2=nn.Conv2d(in_channels=256*3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2d1=nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2d3=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.ps=nn.PixelShuffle(8)
        self.loss = nn.L1Loss()
        self.softmax_0 = nn.Softmax(dim=0)
        self.softmax_1 = nn.Softmax(dim=1)

    def polygons_to_bitmask(self, polygons, height, width):
        if isinstance(polygons, list):
            assert len(polygons) > 0, "COCOAPI does not support empty polygons"
            rles = mask_utils.frPyObjects(polygons, height, width)
            rle = mask_utils.merge(rles)
            return mask_utils.decode(rle).astype(np.uint8)
        if isinstance(polygons, dict):
            h, w = polygons['size']
            rle = mask_utils.frPyObjects(polygons, h, w)
            mask = mask_utils.decode(rle)
            return mask    

    def forward(self,feature,feature_bef,feature_aft,batched_input_cur):
        #print(gt_instance[0].gt_masks[0])
        #print(batched_input_cur[0])

        annos = batched_input_cur[0]["annotations"]
        segms = [obj["segmentation"] for obj in annos]
        gt_masks = [self.polygons_to_bitmask(segm, feature.shape[2]*8, feature.shape[3]*8) for segm in segms]

        gt_mask=np.zeros((1,1,len(gt_masks[0]),len(gt_masks[0][0])))
        for i in range(len(gt_masks)):
            for j in range(len(gt_masks[i])):
                for k in range(len(gt_masks[i][j])):
                    if gt_masks[i][j][k]!=0:
                        gt_mask[0][0][j][k]=1

        feature_bef=torch.sub(feature_bef,feature)
        feature_bef_conv=self.conv2d1(feature_bef)
        feature_aft=torch.sub(feature,feature_aft)
        feature_aft_conv=self.conv2d1(feature_aft)
        
        feat=torch.cat((feature,feature_bef,feature_aft),dim=1)
        feat=self.conv2d2(feat)
        feat=feat*feature_bef_conv+feat*feature_aft_conv+feat
        feat=self.conv2d3(feat)
        out=self.ps(feat)
        # a = torch.ones_like(out,requires_grad=True)
        # b = torch.zeros_like(out,requires_grad=True)
        # out = torch.where(out>0,a,b)
        gt_mask = torch.from_numpy(gt_mask).cuda()
        binary_loss=self.loss(out,gt_mask)
        # print(out.shape)
        # print(gt_mask.shape)
        # exit()
        return out,dict(binary_loss=binary_loss)