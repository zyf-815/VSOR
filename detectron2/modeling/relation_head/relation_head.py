import torch
from .relation_features_extractor import make_relation_features_combine
from .relation_module import make_relation_module
from .saliency_predictors import make_saliency_predictor
from .loss import make_relation_loss_evalutor
from detectron2.utils.registry import Registry

RELATION_REGISTRY = Registry("RELATION_HEAD")


def build_relation_head(cfg, in_channels):
    relation_head_name = cfg.MODEL.RELATION_HEAD.NAME
    relation_head = RELATION_REGISTRY.get(relation_head_name)(cfg, in_channels)
    return relation_head


@RELATION_REGISTRY.register()
class RelationHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(RelationHead, self).__init__()
        self.relation_feature_extractor = make_relation_features_combine(cfg, in_channels)
        relation_feature_dim = cfg.MODEL.RELATION_HEAD.MLP_HEAD_DIM
        self.relation_process = make_relation_module(cfg, relation_feature_dim, 'RelationModule')
        self.saliency_predictor = make_saliency_predictor(cfg)
        self.loss_evalutor = make_relation_loss_evalutor()

    def forward(self, feature, proposals, person_features,feature_bef, proposals_bef, person_features_bef,feature_aft, proposals_aft, person_features_aft):
        if self.training:
            image_sizes = [p.image_size for p in proposals]
            image_sizes_bef = [p.image_size for p in proposals_bef]
            image_sizes_aft = [p.image_size for p in proposals_aft]

            boxes, _, person_probs, gt_ranks = select_from_proposals(proposals, person_features)
            boxes_bef, _, person_probs_bef, gt_ranks_bef = select_from_proposals(proposals_bef, person_features_bef)
            boxes_aft, _, person_probs_aft, gt_ranks_aft = select_from_proposals(proposals_aft, person_features_aft)
            
            features_pre_relation = self.relation_feature_extractor(feature, boxes, image_sizes)
            features_pre_relation_bef = self.relation_feature_extractor(feature_bef, boxes_bef, image_sizes_bef)
            features_pre_relation_aft = self.relation_feature_extractor(feature_aft, boxes_aft, image_sizes_aft)

            features_pre_relation_bef_local = self.relation_feature_extractor(feature_bef, boxes, image_sizes_bef)
            features_pre_relation_aft_local = self.relation_feature_extractor(feature_aft, boxes, image_sizes_aft)

            # print(type(features_pre_relation))  #list 
            # print(len(features_pre_relation))   #1
            # print(len(features_pre_relation[0]))   #3 
            # print(features_pre_relation[0][0].shape)  # the same as the number of instance
            # exit()
            
            features_post_relation = self.relation_process(features_pre_relation, person_probs, person_features,
                                                           features_pre_relation_bef, person_probs_bef, person_features_bef,
                                                           features_pre_relation_aft, person_probs_aft, person_features_aft,
                                                           features_pre_relation_bef_local,features_pre_relation_aft_local)
            
            #print(features_post_relation[0].shape)
            #exit()
            saliency_score = self.saliency_predictor(features_post_relation)
            #print(saliency_score)
            relation_loss = self.loss_evalutor(gt_ranks, saliency_score)
            
            return saliency_score, dict(relation_loss=relation_loss)
        else:
            image_sizes = [p.image_size for p in proposals]
            image_sizes_bef = [p.image_size for p in proposals_bef]
            image_sizes_aft = [p.image_size for p in proposals_aft]
            boxes = [p.pred_boxes for p in proposals]
            boxes_bef = [p.pred_boxes for p in proposals_bef]
            boxes_aft = [p.pred_boxes for p in proposals_aft]
            person_probs = [p.person_probs.view(-1, 1) for p in proposals]
            person_probs_bef = [p.person_probs.view(-1, 1) for p in proposals_bef]
            person_probs_aft = [p.person_probs.view(-1, 1) for p in proposals_aft]

            features_pre_relation = self.relation_feature_extractor(feature, boxes, image_sizes)
            features_pre_relation_bef = self.relation_feature_extractor(feature_bef, boxes_bef, image_sizes_bef)
            features_pre_relation_aft = self.relation_feature_extractor(feature_aft, boxes_aft, image_sizes_aft)

            features_pre_relation_bef_local = self.relation_feature_extractor(feature_bef, boxes, image_sizes_bef)
            features_pre_relation_aft_local = self.relation_feature_extractor(feature_aft, boxes, image_sizes_aft)

            features_post_relation = self.relation_process(features_pre_relation, person_probs, person_features,
                                                           features_pre_relation_bef, person_probs_bef, person_features_bef,
                                                           features_pre_relation_aft, person_probs_aft, person_features_aft,
                                                           features_pre_relation_bef_local,features_pre_relation_aft_local)
            
            saliency_score = self.saliency_predictor(features_post_relation)
            
        return saliency_score, {}


def select_from_proposals(proposals, person_features):
    boxes = []
    result_features = []
    person_probs = []
    gt_ranks = []
    for proposals_per_img, person_features_per_img in zip(proposals, person_features):
        gt_index = torch.nonzero(proposals_per_img.is_gt == 1)
        
        
        boxes.append(proposals_per_img.gt_boxes[gt_index.squeeze()])
        
        result_features.append(person_features_per_img[gt_index.squeeze()])
        person_probs.append(proposals_per_img.person_prob[gt_index])
        gt_ranks.append(proposals_per_img.gt_ranks[gt_index.squeeze()])
    return boxes, result_features, person_probs, gt_ranks
