#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
import os

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging

from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    build_davis_test_loader
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
import cv2
import numpy as np
import copy
logger = logging.getLogger("detectron2")


def do_inference(cfg, model,frames):
    res=[]
    err=0
    with torch.no_grad():
        for i in range(len(frames)):
            image_shape=(frames[i].shape[0],frames[i].shape[1])
            dit2={"image":torch.as_tensor(np.ascontiguousarray(frames[i].transpose(2, 0, 1)))}
            if i==0 :
                dit1={"image":torch.as_tensor(np.ascontiguousarray(frames[i].transpose(2, 0, 1)))}
            else:
                dit1={"image":torch.as_tensor(np.ascontiguousarray(frames[i-1].transpose(2, 0, 1)))}
            if i==len(frames)-1:
                dit3={"image":torch.as_tensor(np.ascontiguousarray(frames[i].transpose(2, 0, 1)))}
            else:
                dit3={"image":torch.as_tensor(np.ascontiguousarray(frames[i+1].transpose(2, 0, 1)))}
            inputs = [[dit1,dit2,dit3]]
            outputs = model(inputs)

            try:
                pred_boxes = outputs["roi_results"][0].pred_boxes
                pred_boxes = pred_boxes.tensor.cpu().data.numpy()
                scores = outputs["roi_results"][0].scores
                scores = scores.cpu().data.numpy()
                pred_masks = outputs["roi_results"][0].pred_masks
                pred_masks = pred_masks.cpu().data.numpy()
                saliency_rank = outputs["rank_result"][0].cpu().data.numpy()
            except:
                err+=1
                all_segmaps = np.zeros(image_shape, dtype=np.float)
                for line in range(len(all_segmaps)//2-10,len(all_segmaps)//2+10):
                    for col in range(len(all_segmaps[line])//2-10,len(all_segmaps[line])//2+10):
                        all_segmaps[line][col]=255
                res.append(all_segmaps)
                continue

            pred = {}
            pred["pred_boxes"] = pred_boxes
            pred["pred_masks"] = pred_masks
            pred["saliency_rank"] = saliency_rank
            threshold=0.6
            # if len(scores)>=3:
            #     threshold=0.8
            # else:
            #     threshold=0.5
            keep = scores > threshold
            pred_boxes = pred_boxes[keep, :]
            pred_masks = pred_masks[keep, :, :]
            saliency_rank = saliency_rank[keep]
            
            segmaps = np.zeros([len(pred_masks), image_shape[0], image_shape[1]])

            for j in range(len(pred_masks)):
                x0 = int(pred_boxes[j, 0])
                y0 = int(pred_boxes[j, 1])
                x1 = int(pred_boxes[j, 2])
                y1 = int(pred_boxes[j, 3])

                segmap = pred_masks[j, 0, :, :]
                
                if (x1-x0)==0:
                    x1=x1+1
                if (y1-y0)==0:
                    y1=y1+1
                    
                segmap = cv2.resize(segmap, (x1-x0, y1-y0),
                                interpolation=cv2.INTER_LANCZOS4)
                segmaps[j, y0:y1, x0:x1] = segmap

            segmaps1 = copy.deepcopy(segmaps)
            all_segmaps = np.zeros(image_shape, dtype=np.float)
            if len(pred_masks) != 0:
                color_index = [sorted(saliency_rank).index(a) + 1 for a in saliency_rank]
                color = [255. / len(saliency_rank) * a for a in color_index]
                cover_region = all_segmaps != 0
                for k in range(len(segmaps1), 0, -1):
                    obj_id = color_index.index(k)
                    seg = segmaps1[obj_id]
                    seg[seg >= 0.5] = color[obj_id]
                    seg[seg < 0.5] = 0
                    seg[cover_region] = 0
                    all_segmaps += seg
                    cover_region = all_segmaps != 0
                all_segmaps = all_segmaps.astype(np.int)
            if np.sum(all_segmaps)==0:
                for line in range(len(all_segmaps)//2-10,len(all_segmaps)//2+10):
                    for col in range(len(all_segmaps[line])//2-10,len(all_segmaps[line])//2+10):
                        all_segmaps[line][col]=255
            res.append(all_segmaps)
            # cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/retarget/'+str(0)+'.png', all_segmaps)
            # cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/retarget/'+str(0)+'.jpg', frames[i])
            # print('\r{}/{}'.format(i, len(frames)), end="", flush=True)
    res=np.array(res)
    res=res.transpose(1,2,0)
    print('err_find:',err)
    # exit()
    return res

def do_test(cfg, model,video_path):
    video = cv2.VideoCapture(video_path)
    frames=[]
    while True:
        _, frame = video.read()
        if frame is None:
            break
        frames.append(frame)
    model.eval()
    with torch.no_grad():
        for i in range(len(frames)):
            image_shape=(frames[i].shape[0],frames[i].shape[1])
            dit2={"image":torch.as_tensor(np.ascontiguousarray(frames[i].transpose(2, 0, 1)))}
            if i==0 :
                dit1={"image":torch.as_tensor(np.ascontiguousarray(frames[i].transpose(2, 0, 1)))}
            else:
                dit1={"image":torch.as_tensor(np.ascontiguousarray(frames[i-1].transpose(2, 0, 1)))}
            if i==len(frames)-1:
                dit3={"image":torch.as_tensor(np.ascontiguousarray(frames[i].transpose(2, 0, 1)))}
            else:
                dit3={"image":torch.as_tensor(np.ascontiguousarray(frames[i+1].transpose(2, 0, 1)))}
            inputs = [[dit1,dit2,dit3]]
            outputs = model(inputs)

            try:
                pred_boxes = outputs["roi_results"][0].pred_boxes
                pred_boxes = pred_boxes.tensor.cpu().data.numpy()
                scores = outputs["roi_results"][0].scores
                scores = scores.cpu().data.numpy()
                pred_masks = outputs["roi_results"][0].pred_masks
                pred_masks = pred_masks.cpu().data.numpy()
                saliency_rank = outputs["rank_result"][0].cpu().data.numpy()
            except:
                all_segmaps = np.zeros(image_shape, dtype=np.float)
                for line in range(len(all_segmaps)//2-10,len(all_segmaps)//2+10):
                    for col in range(len(all_segmaps[line])//2-10,len(all_segmaps[line])//2+10):
                        all_segmaps[line][col]=255
                cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/retarget/'+str(1)+'.png', all_segmaps)
                cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/retarget/'+str(1)+'.jpg', frames[i])
                continue

            pred = {}
            pred["pred_boxes"] = pred_boxes
            pred["pred_masks"] = pred_masks
            pred["saliency_rank"] = saliency_rank
            threshold=0.8
            keep = scores > threshold
            pred_boxes = pred_boxes[keep, :]
            pred_masks = pred_masks[keep, :, :]
            saliency_rank = saliency_rank[keep]
            
            segmaps = np.zeros([len(pred_masks), image_shape[0], image_shape[1]])

            for j in range(len(pred_masks)):
                x0 = int(pred_boxes[j, 0])
                y0 = int(pred_boxes[j, 1])
                x1 = int(pred_boxes[j, 2])
                y1 = int(pred_boxes[j, 3])

                segmap = pred_masks[j, 0, :, :]
                
                if (x1-x0)==0:
                    x1=x1+1
                if (y1-y0)==0:
                    y1=y1+1
                    
                segmap = cv2.resize(segmap, (x1-x0, y1-y0),
                                interpolation=cv2.INTER_LANCZOS4)
                segmaps[j, y0:y1, x0:x1] = segmap

            segmaps1 = copy.deepcopy(segmaps)
            all_segmaps = np.zeros(image_shape, dtype=np.float)
            if len(pred_masks) != 0:
                color_index = [sorted(saliency_rank).index(a) + 1 for a in saliency_rank]
                color = [255. / len(saliency_rank) * a for a in color_index]
                cover_region = all_segmaps != 0
                for k in range(len(segmaps1), 0, -1):
                    obj_id = color_index.index(k)
                    seg = segmaps1[obj_id]
                    seg[seg >= 0.5] = color[obj_id]
                    seg[seg < 0.5] = 0
                    seg[cover_region] = 0
                    all_segmaps += seg
                    cover_region = all_segmaps != 0
                all_segmaps = all_segmaps.astype(np.int)
            if np.sum(all_segmaps)==0:
                for line in range(len(all_segmaps)//2-10,len(all_segmaps)//2+10):
                    for col in range(len(all_segmaps[line])//2-10,len(all_segmaps[line])//2+10):
                        all_segmaps[line][col]=255
            cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/retarget/'+str(1)+'.png', all_segmaps)
            cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/retarget/'+str(1)+'.jpg', frames[i])
            print('\r{}/{}'.format(i, len(frames)), end="", flush=True)
    return 0

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    model_root_dir = "./Rank_Saliency/Models/RVSOR(52)"
    model_names = 'model_0019999.pth'

    model_dir = os.path.join(model_root_dir, model_names)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        model_dir, resume=args.resume)
    out = do_test(cfg, model,'/home/zyf/code/RetargetVid-main/DHF1k/010.AVI')

if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
