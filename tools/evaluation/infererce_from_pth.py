# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import Counter
import time
import os
from contextlib import contextmanager
import torch
from tqdm import tqdm
import numpy as np
import copy
import cv2
from spearman_correlation import evalu as rank_evalu
from mae_fmeasure_2 import evalu as mf_evalu
from detectron2.data.davis import davis_val
import pickle as pkl
from collections import Counter
from skimage import measure
import numpy as np
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from PIL import Image, ImageOps
from skimage import io, transform
from sklearn.metrics import mean_absolute_error


def inference():
    res = []

    dataset = davis_val()
    no_figure=0
    
    pth2='/home/zyf/code/SOR-main/sor_ppa/output/sor/transformer-pos_twostream_dense_1/inference/instances_predictions.pth'
    preds= torch.load(pth2)
    
    # print()
    # print(preds[25]['predictions_for_ranking'].pred_ranks)
    pred_dict={}
    for i in range(len(preds)):
        if len(preds[i]['instances'])==0:
            continue
        try:
            pred_dict[preds[i]['file_name']]=[preds[i]['predictions_for_ranking'].pred_masks,preds[i]['predictions_for_ranking'].pred_ranks]
        except:
            print(preds[i])
            exit()
    del preds
    
    
    
    for i in range(len(dataset)):
    #for i in range(1):
        try:
            inputs = [dataset[i]]

            gt_boxes = inputs[0][1]["gt_boxes"]
            gt_masks = inputs[0][1]["gt_masks"]
            gt_ranks = inputs[0][1]["gt_rank"]
            image_shape = inputs[0][1]["image_shape"]
            name=inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')

            if inputs[0][1]["file_name"] in pred_dict:
                segmaps_=np.asarray(pred_dict[inputs[0][1]["file_name"]][0]).astype(int)
                rank_scores_=np.asarray(pred_dict[inputs[0][1]["file_name"]][1]).astype(int)
                segmaps=[]
                rank_scores=[]
                for j in range(len(rank_scores_)):
                    if rank_scores_[j]<0:
                        continue
                    segmaps.append(segmaps_[j])
                    rank_scores.append(rank_scores_[j])
                segmaps=np.array(segmaps)
                rank_scores=np.array(rank_scores)
            else:
                exit()
            # exit()
            
            res.append({'gt_masks': gt_masks, 'segmaps': segmaps, 'gt_ranks': gt_ranks,
                        'rank_scores': rank_scores, 'img_name': name})

            print('\r{}/{}'.format(i, len(dataset)), end="", flush=True)
        except:
            no_figure+=1
            print(name)
            continue
    #return 0
    print('wrong figure=',no_figure)
    r_corre = rank_evalu(res, 0.5)
    r_f = mf_evalu(res)
    print('r_corre:',r_corre)
    print('r_f:',r_f)
    return r_corre, r_f


if __name__ == '__main__':
    inference()
    pass