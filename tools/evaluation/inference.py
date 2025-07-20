# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import time
import os
from contextlib import contextmanager
import torch
from tqdm import tqdm
import numpy as np
import copy
import cv2
from .spearman_correlation import evalu as rank_evalu
from .mae_fmeasure_2 import evalu as mf_evalu
from detectron2.data.davis import davis_val
import pickle as pkl
from collections import Counter


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def inference(cfg, model,draw=False):
    res = []
    res_mae=[]
    # out_dir = '/home/lilong/project/rank_saliency/detectron2/' \
    #           '19_origin_relation_local_new_global_add_person_feature_noprobs_inrela_beta_2/tools/predictions'
    dataset = davis_val(cfg, False)
    with inference_context(model), torch.no_grad():
        #for i in range(5):
        no_figure=0
        for i in range(len(dataset)):
            try:
                inputs = [dataset[i]]
                outputs = model(inputs)
                # print(inputs[0][1]['file_name'])
                # print(outputs["roi_results"][0].pred_boxes.tensor.cpu().data.numpy())
                # print(outputs["roi_results"][0].scores.cpu().data.numpy())
                # print(outputs)
                # exit()
                
                # if i==242:
                #     print(inputs[0][1]["file_name"])
                #     break

                gt_boxes = inputs[0][1]["gt_boxes"]
                gt_masks = inputs[0][1]["gt_masks"]
                gt_ranks = inputs[0][1]["gt_rank"]
                name = inputs[0][1]["file_name"].split('/')[-1][:-4]
                image_shape = inputs[0][1]["image_shape"]
                #binary_mask=outputs['binary_result'].squeeze(0).squeeze(0)
                 
                pred_boxes = outputs["roi_results"][0].pred_boxes
                pred_boxes = pred_boxes.tensor.cpu().data.numpy()
                scores = outputs["roi_results"][0].scores
                scores = scores.cpu().data.numpy()
                pred_masks = outputs["roi_results"][0].pred_masks
                pred_masks = pred_masks.cpu().data.numpy()

                saliency_rank = outputs["rank_result"][0].cpu().data.numpy()

                

                pred = {}
                pred["pred_boxes"] = pred_boxes
                pred["pred_masks"] = pred_masks
                pred["saliency_rank"] = saliency_rank

                # with open(os.path.join(out_dir, name), 'wb') as f:
                #     pkl.dump(pred, f)

                # print('./draw_res/'+name+'.jpg')
                # print(scores)
                # print(saliency_rank)

                #only keep the box and masks which have score>0.6
                AVE=sum(scores)/len(scores)
                threshold=0.6
                keep = scores > threshold
                pred_boxes = pred_boxes[keep, :]
                pred_masks = pred_masks[keep, :, :]
                saliency_rank = saliency_rank[keep]

                # threshold_rank=1.5
                # keep_rank = saliency_rank > threshold_rank
                # keep_rank=keep_rank.flatten()
                # pred_boxes = pred_boxes[keep_rank, :]
                # pred_masks = pred_masks[keep_rank, :, :]
                # saliency_rank = saliency_rank[keep_rank]

                image_shape = inputs[0][1]["image_shape"]
                name = inputs[0][1]["file_name"].split('/')[-1]

                # print(name)

                # if name != 'COCO_val2014_000000095297.jpg':
                #     continue

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

                res.append({'gt_masks': gt_masks, 'segmaps': segmaps, 'scores': scores, 'gt_ranks': gt_ranks,
                            'rank_scores': saliency_rank, 'img_name': name,'gt_boxes':gt_boxes,'pred_box':pred["pred_boxes"]})
                res_mae.append({'gt_masks': gt_masks, 'segmaps': segmaps, 'scores': scores, 'gt_ranks': gt_ranks,
                            'rank_scores': saliency_rank, 'img_name': name,'gt_boxes':gt_boxes,'pred_box':pred["pred_boxes"]})
                #print(saliency_rank)


                # if i==20:
                #     break
                # segmaps1 = copy.deepcopy(segmaps)
                # all_segmaps = np.zeros_like(gt_masks[0], dtype=np.float)
                # if len(pred_masks) != 0:
                #     color_index = [sorted(saliency_rank).index(a) + 1 for a in saliency_rank]
                #     color = [255. / len(saliency_rank) * a for a in color_index]
                #     cover_region = all_segmaps != 0
                #     for i in range(len(segmaps1), 0, -1):
                #         obj_id = color_index.index(i)
                #         seg = segmaps1[obj_id]
                #         seg[seg >= 0.5] = color[obj_id]
                #         seg[seg < 0.5] = 0
                #         seg[cover_region] = 0
                #         all_segmaps += seg
                #         cover_region = all_segmaps != 0
                #     all_segmaps = all_segmaps.astype(np.int)
                # cv2.imwrite('./saliency_maps/{}.png'.format(name[:-4]), all_segmaps)

                # print(len(res))
                print('\r{}/{}'.format(len(res), len(dataset)), end="", flush=True)
                
                #print(len(gt_boxes))

                if draw:
                    img_name=inputs[0][1]['file_name']
                    img_draw_gt_boxes = cv2.imread(img_name)
                    img_draw_pre_boxes = cv2.imread(img_name)
                    box_color = (255,0,255)   

                    for j in range(len(gt_boxes)):
                        x0 = int(gt_boxes[j, 0])
                        y0 = int(gt_boxes[j, 1])
                        x1 = int(gt_boxes[j, 2])
                        y1 = int(gt_boxes[j, 3])
                    
                        cv2.rectangle(img_draw_gt_boxes, (x0,y0), (x1,y1), color=box_color, thickness=2)
     
                    for j in range(len(pred_masks)):
                        x0 = int(pred_boxes[j, 0])
                        y0 = int(pred_boxes[j, 1])
                        x1 = int(pred_boxes[j, 2])
                        y1 = int(pred_boxes[j, 3])
                    
                        cv2.rectangle(img_draw_pre_boxes, (x0,y0), (x1,y1), color=box_color, thickness=2)
                    
                    imgs_c=np.hstack([img_draw_gt_boxes,img_draw_pre_boxes])
                    cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/draw_box/'+img_name.split('/')[-1], imgs_c)

            #if draw and len(gt_boxes)>1:

                    img_name=inputs[0][1]['file_name']
                    img_draw_gt_boxes = cv2.imread(img_name)
                    img_draw_pre_boxes = cv2.imread(img_name)
                    box_color = (255,0,255)
                    img_name2=img_name.replace('img','ranking saliency masks/img')
                    gt_mask_img=cv2.imread(img_name2.replace('jpg','png'),0)
                
                    gt_map = np.zeros((image_shape[0],image_shape[1]))
                    gt_index = (np.asarray(gt_ranks) + 1).astype(np.float)
                    color = [255. / len(gt_ranks) * a for a in gt_index]
                    for i in range(len(gt_masks)):
                        gt_map[gt_masks[i] != 0] = color[i]
                        #gt_map[gt_masks[i] != 0] = 255.0

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
                    all_segmaps = np.zeros_like(gt_masks[0], dtype=np.float)
                    if len(pred_masks) != 0:
                        color_index = [sorted(saliency_rank).index(a) + 1 for a in saliency_rank]
                        color = [255. / len(saliency_rank) * a for a in color_index]
                        cover_region = all_segmaps != 0
                        for k in range(len(segmaps1), 0, -1):
                            obj_id = color_index.index(k)
                            seg = segmaps1[obj_id]
                            seg[seg >= 0.5] = color[obj_id]
                            #seg[seg >= 0.5] = 255.0
                            seg[seg < 0.5] = 0
                            seg[cover_region] = 0
                            all_segmaps += seg
                            cover_region = all_segmaps != 0
                        all_segmaps = all_segmaps.astype(np.int)
                    #cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/draw_mask/'+str(i)+'.jpg', all_segmaps)

                    imgs_d=np.hstack([gt_map,all_segmaps])           
                    cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/draw_mask/'+img_name.split('/')[-1].replace('jpg','png'), all_segmaps)

                    # imgs_r=np.vstack([imgs_c,imgs_d])
                    # cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/draw_res_epoch1/'+img_name.split('/')[-1], imgs_r)

            except:
                no_figure+=1
                print(inputs[0][1]["file_name"].split('/')[-1])

                # res_mae.append({'gt_masks': gt_masks, 'segmaps': np.zeros([len(pred_masks), image_shape[0], image_shape[1]]), 'scores': [0], 'gt_ranks': gt_ranks,
                #             'rank_scores': [0], 'img_name': name,'gt_boxes':gt_boxes})
                continue
        # lst=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        # for k in lst:
        #     r_corre = rank_evalu(res, k)
        #     print(r_corre)
        print('wrong figure=',no_figure)
        r_corre = rank_evalu(res, 0.5)
        #r_map=mf_map(res)
        r_f = mf_evalu(res_mae)

        return r_corre, r_f,0
        #return 0, {"mae": 0, "f_measure": 0},0
        # return 0, r_f

def mf_map(res,iou_threshold=0.5,num_classes=1):
    pred_bboxes=[]
    idx=0
    for i in range(len(res)):
        for j in range(len(res[i]['scores'])):
            temp=[idx,0,res[i]['scores'][j],res[i]['pred_box'][j][0],res[i]['pred_box'][j][1],res[i]['pred_box'][j][2],res[i]['pred_box'][j][3]]
            pred_bboxes.append(temp)
        idx+=1
    true_boxes=[]
    idx=0
    for i in range(len(res)):
        for j in range(len(res[i]['gt_boxes'])):
            temp=[idx,0,1,res[i]['gt_boxes'][j][0],res[i]['gt_boxes'][j][1],res[i]['gt_boxes'][j][2],res[i]['gt_boxes'][j][3]]
            true_boxes.append(temp)
        idx+=1

    average_precisions=[]#存储每一个类别的AP
    epsilon=1e-6#防止分母为0

    #对于每一个类别
    for c in range(num_classes):

        detections=[]#存储预测为该类别的bbox
        ground_truths=[]#存储本身就是该类别的bbox(GT)
        
        for detection in pred_bboxes:
            if detection[1]==c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1]==c:
                ground_truths.append(true_box)

        amount_bboxes=Counter(gt[0] for gt in ground_truths)

        for key,val in amount_bboxes.items():
            amount_bboxes[key]=torch.zeros(val)#置0，表示这些真实框初始时都没有与任何预测框匹配
        #此时，amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}
        
        #将预测框按照置信度从大到小排序
        detections.sort(key=lambda x:x[2],reverse=True)
        #初始化TP,FP
        TP=torch.zeros(len(detections))
        FP=torch.zeros(len(detections))
        
        #TP+FN就是当前类别GT框的总数，是固定的
        total_true_bboxes=len(ground_truths)
    
        #如果当前类别一个GT框都没有，那么直接跳过即可
        if total_true_bboxes == 0:
            continue
        
        #对于每个预测框，先找到它所在图片中的所有真实框，然后计算预测框与每一个真实框之间的IoU，大于IoU阈值且该真实框没有与其他预测框匹配，则置该预测框的预测结果为TP，否则为FP
        for detection_idx,detection in enumerate(detections):

            ground_truth_img=[bbox for bbox in ground_truths if bbox[0]==detection[0]]
            num_gts=len(ground_truth_img)
            
            best_iou=0
            for idx,gt in enumerate(ground_truth_img):
                #计算当前预测框detection与它所在图片内的每一个真实框的IoU
                iou=insert_over_union(torch.tensor(detection[3:]),torch.tensor(gt[3:]))
                if iou >best_iou:
                    best_iou=iou
                    best_gt_idx=idx
            if best_iou>iou_threshold:
                #这里的detection[0]是amount_bboxes的一个key，表示图片的编号，best_gt_idx是该key对应的value中真实框的下标
                if amount_bboxes[detection[0]][best_gt_idx]==0:#只有没被占用的真实框才能用，0表示未被占用（占用：该真实框与某预测框匹配【两者IoU大于设定的IoU阈值】）
                    TP[detection_idx]=1#该预测框为TP
                    amount_bboxes[detection[0]][best_gt_idx]=1#将该真实框标记为已经用过了，不能再用于其他预测框。因为一个预测框最多只能对应一个真实框（最多：IoU小于IoU阈值时，预测框没有对应的真实框)
                else:
                    FP[detection_idx]=1#虽然该预测框与真实框中的一个框之间的IoU大于IoU阈值，但是这个真实框已经与其他预测框匹配，因此该预测框为FP
            else:
                FP[detection_idx]=1#该预测框与真实框中的每一个框之间的IoU都小于IoU阈值，因此该预测框直接为FP
                
        TP_cumsum=torch.cumsum(TP,dim=0)
        FP_cumsum=torch.cumsum(FP,dim=0)
        
        #套公式
        recalls=TP_cumsum/(total_true_bboxes+epsilon)
        precisions=torch.divide(TP_cumsum,(TP_cumsum+FP_cumsum+epsilon))
        
        #把[0,1]这个点加入其中
        precisions=torch.cat((torch.tensor([1]),precisions))
        recalls=torch.cat((torch.tensor([0]),recalls))
        #使用trapz计算AP
        average_precisions.append(torch.trapz(precisions,recalls))
        
    return sum(average_precisions)/len(average_precisions) 
 
 
def insert_over_union(boxes_preds,boxes_labels):
    
    box1_x1=boxes_preds[...,0:1]
    box1_y1=boxes_preds[...,1:2]
    box1_x2=boxes_preds[...,2:3]
    box1_y2=boxes_preds[...,3:4]#shape:[N,1]
    
    box2_x1=boxes_labels[...,0:1]
    box2_y1=boxes_labels[...,1:2]
    box2_x2=boxes_labels[...,2:3]
    box2_y2=boxes_labels[...,3:4]
    
    x1=torch.max(box1_x1,box2_x1)
    y1=torch.max(box1_y1,box2_y1)
    x2=torch.min(box1_x2,box2_x2)
    y2=torch.min(box1_y2,box2_y2)
    
    
    #计算交集区域面积
    intersection=(x2-x1).clamp(0)*(y2-y1).clamp(0)
    
    box1_area=abs((box1_x2-box1_x1)*(box1_y1-box1_y2))
    box2_area=abs((box2_x2-box2_x1)*(box2_y1-box2_y2))
    
    return intersection/(box1_area+box2_area-intersection+1e-6)