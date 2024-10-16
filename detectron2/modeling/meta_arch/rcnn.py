# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
import copy
from detectron2.structures import ImageList
from detectron2.structures import Instances
import math
from .binary_salient_build import build_Salient_predict
from ..backbone import build_backbone
from ..backbone import build_bottom_up_fuse
from ..relation_head import build_relation_head
from ..backbone.panet_fpn import PAnetFPN
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
__all__ = ["RankSaliencyNetwork"]


@META_ARCH_REGISTRY.register()
class RankSaliencyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.bottom_up_fuse = build_bottom_up_fuse(cfg)

        #self.binaryPredict=build_Salient_predict(cfg)

        self.conv2d1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.conv2d2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=4,stride=2,padding=1)
        self.MaxPool2d = nn.MaxPool2d(2, stride=2,padding=0)
        self.AvgPool2d = nn.AvgPool2d(2, stride=2,padding=0)

        self.panet_fpn = PAnetFPN()
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.relation_head = build_relation_head(cfg, self.bottom_up_fuse.out_channels)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)



    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, batched_inputs,its=0):
        if not self.training:
            return self.inference(batched_inputs)

        # batched_input_cur=batched_inputs[:][1]
        # batched_input_bef=batched_inputs[:][0]
        # batched_input_aft=batched_inputs[:][2]
        batched_input_cur=[temp[1] for temp in batched_inputs]
        batched_input_bef=[temp[0] for temp in batched_inputs]
        batched_input_aft=[temp[2] for temp in batched_inputs]

        #images = self.preprocess_image(batched_inputs)
        
        images = self.preprocess_image(batched_input_cur)
        images_bef = self.preprocess_image(batched_input_bef)
        images_aft = self.preprocess_image(batched_input_aft)

        #gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        gt_instances = [x["instances"].to(self.device) for x in batched_input_cur]
        gt_instances_bef = [x["instances"].to(self.device) for x in batched_input_bef]
        gt_instances_aft = [x["instances"].to(self.device) for x in batched_input_aft]

        features_res, features_fpn = self.backbone(images.tensor)
        features_res_bef, features_fpn_bef = self.backbone(images_bef.tensor)
        features_res_aft, features_fpn_aft = self.backbone(images_aft.tensor)

        features_relation = self.bottom_up_fuse(features_res)
        features_relation_bef = self.bottom_up_fuse(features_res_bef)
        features_relation_aft = self.bottom_up_fuse(features_res_aft)

        # binary_mask,binary_loss=self.binaryPredict(features_relation,features_relation_bef,features_relation_aft,batched_input_cur)
        # binary_mask_bef,binary_loss_bef=self.binaryPredict(features_relation_bef,features_relation_bef,features_relation,batched_input_cur)
        # binary_mask_aft,binary_loss_aft=self.binaryPredict(features_relation_aft,features_relation,features_relation_aft,batched_input_cur)
        
        features = self.panet_fpn(features_fpn)
        features_bef = self.panet_fpn(features_fpn_bef)
        features_aft = self.panet_fpn(features_fpn_aft)
        
        # features_salient = self.panet_fpn(features_fpn)
        # features_salient_bef = self.panet_fpn(features_fpn_bef)
        # features_salient_aft = self.panet_fpn(features_fpn_aft)

        # binary_mask= self.conv2d1(binary_mask)
        # binary_mask= self.conv2d2(binary_mask)
        # features_salient['p2']=torch.mul(features['p2'],binary_mask)+features['p2']
        # binary_mask= self.MaxPool2d(binary_mask)
        # features_salient['p3']=torch.mul(features['p3'],binary_mask)+features['p3']

        # binary_mask_bef= self.conv2d1(binary_mask_bef)
        # binary_mask_bef= self.conv2d2(binary_mask_bef)
        # features_salient_bef['p2']=torch.mul(features_bef['p2'],binary_mask_bef)+features_bef['p2']
        # binary_mask_bef= self.MaxPool2d(binary_mask_bef)
        # features_salient_bef['p3']=torch.mul(features_bef['p3'],binary_mask_bef)+features_bef['p3']

        # binary_mask_aft= self.conv2d1(binary_mask_aft)
        # binary_mask_aft= self.conv2d2(binary_mask_aft)
        # features_salient_aft['p2']=torch.mul(features_aft['p2'],binary_mask_aft)+features_aft['p2']
        # binary_mask_aft= self.MaxPool2d(binary_mask_aft)
        # features_salient_aft['p3']=torch.mul(features_aft['p3'],binary_mask_aft)+features_aft['p3']

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposals_bef, _ = self.proposal_generator(images_bef, features_bef, gt_instances_bef)
        proposals_aft, _ = self.proposal_generator(images_aft, features_aft, gt_instances_aft)
        results, detector_losses, person_features = self.roi_heads(images, features, proposals, gt_instances)
        results_bef, _, person_features_bef = self.roi_heads(images_bef, features_bef, proposals_bef, gt_instances_bef)
        results_aft, _, person_features_aft = self.roi_heads(images_aft, features_aft, proposals_aft, gt_instances_aft)

        _, relation_loss = self.relation_head(features_relation, results, person_features,features_relation_bef, results_bef, person_features_bef,features_relation_aft, results_aft, person_features_aft)

        #print(relation_loss)
        
        # losses1=sum(relation_loss.values())
        # if not torch.isfinite(losses1).all():
        #     relation_loss['relation_loss']=torch.tensor(1, device='cuda:0')
        losses = {}
        #losses.update(binary_loss)
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(relation_loss)
        
        #losses1 = sum(losses.values())
        #print()
        
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        
        # batched_input_cur=[batched_inputs[0][1]]
        # batched_input_bef=[batched_inputs[0][0]]
        # batched_input_aft=[batched_inputs[0][2]]
        batched_input_cur=[temp[1] for temp in batched_inputs]
        batched_input_bef=[temp[0] for temp in batched_inputs]
        batched_input_aft=[temp[2] for temp in batched_inputs]

        images = self.preprocess_image(batched_input_cur)
        images_bef = self.preprocess_image(batched_input_bef)
        images_aft = self.preprocess_image(batched_input_aft)
        # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features_res, features_fpn = self.backbone(images.tensor)
        features_res_bef, features_fpn_bef = self.backbone(images_bef.tensor)
        features_res_aft, features_fpn_aft = self.backbone(images_aft.tensor)
        features_relation = self.bottom_up_fuse(features_res)
        features_relation_bef = self.bottom_up_fuse(features_res_bef)
        features_relation_aft = self.bottom_up_fuse(features_res_aft)

        # binary_mask,binary_loss=self.binaryPredict(features_relation,features_relation_bef,features_relation_aft,batched_input_cur)
        # binary_mask_bef,binary_loss_bef=self.binaryPredict(features_relation_bef,features_relation_bef,features_relation,batched_input_cur)
        # binary_mask_aft,binary_loss_aft=self.binaryPredict(features_relation_aft,features_relation,features_relation_aft,batched_input_cur)

        features = self.panet_fpn(features_fpn)
        features_bef = self.panet_fpn(features_fpn_bef)
        features_aft = self.panet_fpn(features_fpn_aft)

        # features_salient = self.panet_fpn(features_fpn)
        # features_salient_bef = self.panet_fpn(features_fpn_bef)
        # features_salient_aft = self.panet_fpn(features_fpn_aft)

        # binary_mask= self.conv2d1(binary_mask)
        # binary_mask= self.conv2d2(binary_mask)
        # features_salient['p2']=torch.mul(features['p2'],binary_mask)+features['p2']
        # binary_mask= self.MaxPool2d(binary_mask)
        # features_salient['p3']=torch.mul(features['p3'],binary_mask)+features['p3']

        # binary_mask_bef= self.conv2d1(binary_mask_bef)
        # binary_mask_bef= self.conv2d2(binary_mask_bef)
        # features_salient_bef['p2']=torch.mul(features_bef['p2'],binary_mask_bef)+features_bef['p2']
        # binary_mask_bef= self.MaxPool2d(binary_mask_bef)
        # features_salient_bef['p3']=torch.mul(features_bef['p3'],binary_mask_bef)+features_bef['p3']

        # binary_mask_aft= self.conv2d1(binary_mask_aft)
        # binary_mask_aft= self.conv2d2(binary_mask_aft)
        # features_salient_aft['p2']=torch.mul(features_aft['p2'],binary_mask_aft)+features_aft['p2']
        # binary_mask_aft= self.MaxPool2d(binary_mask_aft)
        # features_salient_aft['p3']=torch.mul(features_aft['p3'],binary_mask_aft)+features_aft['p3']

        proposals, _ = self.proposal_generator(images, features, None)
        proposals_bef, _ = self.proposal_generator(images_bef, features_bef, None)
        proposals_aft, _ = self.proposal_generator(images_aft, features_aft, None)
        results, _, person_features = self.roi_heads(images, features, proposals)
        results_bef, _, person_features_bef = self.roi_heads(images_bef, features_bef, proposals_bef)
        results_aft, _, person_features_aft = self.roi_heads(images_aft, features_aft, proposals_aft)
        
        salienct_rank, _ = self.relation_head(features_relation, results, person_features,
                                           features_relation_bef, results_bef, person_features_bef,
                                           features_relation_aft, results_aft, person_features_aft)

    #'binary_result' : binary_mask,
        return {
                'roi_results': results,
                'rank_result': salienct_rank}
    #'rank_result': salienct_rank

def generate_gt_proposals_single_image(gt_boxes, device, image_size):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    # Concatenating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))

    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)
    gt_proposal = Instances(image_size)

    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    gt_proposal.is_gt = torch.ones(len(gt_boxes), device=device)
    return gt_proposal
