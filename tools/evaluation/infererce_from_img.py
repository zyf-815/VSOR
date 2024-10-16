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

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
    stats = stats[stats[:,4].argsort()]
    return stats[:-1] # 排除最外层的连通图

def inference():
    res = []

    dataset = davis_val()
    no_figure=0
    for i in range(len(dataset)):
    #for i in range(1):
        try:
            inputs = [dataset[i]]

            gt_boxes = inputs[0][1]["gt_boxes"]
            gt_masks = inputs[0][1]["gt_masks"]
            gt_ranks = inputs[0][1]["gt_rank"]
            image_shape = inputs[0][1]["image_shape"]
            name=inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')

            #pre_fig_root='/home/zyf/code/Saliency-Ranking-main/tools/preds_RVSOD_model_0009999/preds_RVSOD_model_0009999/'
            # pre_fig=pre_fig_root+inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')
            # image2 = Image.open(pre_fig)

            # pre_fig_root='/home/zyf/code/Saliency-Ranking-main/tools/preds_RVSOD_model/preds_RVSOD_model_0009999_resize_2/'
            # pre_fig=pre_fig_root+inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')

            pre_fig_root='/data2/zyf/VSOR_old/preds_RVSOD_model/preds_RVSOD_model_0009999/'
            pre_fig=pre_fig_root+inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')

            #print(name)
            # if name.split('_')[0] == 'actioncliptrain00478':
            #     print(i)
            # else:
            #     continue
            image = Image.open(pre_fig)

            size = (int(image_shape[1]), int(image_shape[0]))
            if size!=image.size:
                image=image.resize(size,Image.NEAREST)
            # image.save('/home/zyf/code/Saliency-Ranking-main/tools/preds_RVSOD_model_0009999/preds_RVSOD_model_0009999_resize_2/'+name)
            # print('\r{}/{}'.format(i, len(dataset)), end="", flush=True)
            # continue

            maskdata=np.array(image.convert('L'))

            num_class=[]
            for j in range(len(maskdata)):
                for k in range(len(maskdata[j])):
                    if maskdata[j][k]!=0 and maskdata[j][k] not in num_class:
                        num_class.append(maskdata[j][k])
            #if len(num_class)>5:
            #if True: 
                #print(num_class)
                #no_figure+=1
                #print(name)
                #continue
            annotations_sm=[]
            greydepth=[]
            for class_idx in num_class:
                cur_mask=copy.deepcopy(maskdata)
                dit={}

                for j in range(len(maskdata)):
                    for k in range(len(maskdata[j])):
                        if maskdata[j][k]==class_idx:
                            cur_mask[j][k]=1
                            ok=[j,k]
                        else:
                            cur_mask[j][k]=0

                greydepth.append(maskdata[ok[0]][ok[1]])
                
                dit['segmentation']=cur_mask
                bboxs = mask_find_bboxs(cur_mask)[0]

                dit['bbox']=(bboxs[0],bboxs[1],bboxs[2],bboxs[3])
                dit['bbox_mode']='xywh'
                dit['is_person']=0
        
                annotations_sm.append(dit)
            rank_sm=[]
            for j in range(len(greydepth)):
                cur=0
                for k in greydepth:
                    if greydepth[j]>k:
                        cur+=1
                rank_sm.append(cur)

        
            segmaps = np.zeros([len(rank_sm), image_shape[0], image_shape[1]])

            for j in range(len(rank_sm)):
                segmaps[j, :, :]=annotations_sm[j]['segmentation']

            rank_scores=[]
            for j in rank_sm:
                rank_scores.append([j+5])
            rank_scores=np.array(rank_scores)
            # color=255.
            # for temp in range(len(annotations_sm)):
            #     for j in range(len(annotations_sm[temp]['segmentation'])):
            #         for k in range(len(annotations_sm[temp]['segmentation'][j])):
            #             if annotations_sm[temp]['segmentation'][j][k]==1:
            #                 annotations_sm[temp]['segmentation'][j][k]=color
            #     cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/evaluation/res/'+str(num_class[temp])+'.jpg', annotations_sm[temp]['segmentation'])

            #print(len(segmaps[0][0]))
            ### gt_masks   N*W*H

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

def eval_mae():

    dataset = davis_val()
    no_figure=0
    mae_sum=0
    num=0
    for i in range(len(dataset)):
    #for i in range(1):
        #try:
            inputs = [dataset[i]]
            gt_name=inputs[0][1]["file_name"].replace('/img/','/ranking saliency masks/img/').replace('.jpg','.png')

            name=inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')

            pre_fig_root='/home/zyf/code/Saliency-Ranking-main/tools/preds_RVSOD_model_0009999/preds_RVSOD_model_0009999_resize_2/'
            pre_fig=pre_fig_root+inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')
            image = Image.open(pre_fig)
            maskdata=np.array(image.convert('L'))
            for j in range(len(maskdata)):
                for k in range(len(maskdata[j])):
                    if maskdata[j][k]!=0:
                        maskdata[j][k]=1
            
            #print(gt_name)
            image_gt = Image.open(gt_name)
            maskdata_gt=np.array(image_gt.convert('L'))
            for j in range(len(maskdata_gt)):
                for k in range(len(maskdata_gt[j])):
                    if maskdata_gt[j][k]!=0:
                        maskdata_gt[j][k]=1
            
            maskdata=maskdata.reshape(-1)
            maskdata_gt=maskdata_gt.reshape(-1)
            
            if len(maskdata_gt)==len(maskdata):
                mae=0
                for k in range(len(maskdata_gt)):
                    if maskdata_gt[k]!=maskdata[k]:
                        mae+=1
                mae=mae/len(maskdata)
                mae_sum+=mae
            else:
                no_figure+=1
                continue

            print('\r{}/{}'.format(i, len(dataset)), end="", flush=True)
            num+=1
        # except:
        #     print(name)
        #     no_figure+=1
        #     continue

    print(mae_sum/len(dataset))
    print('no_figure:',no_figure)

def resize():

    dataset = davis_val()
    no_figure=0
    
    num=0
    mae_sum=0
    for i in range(len(dataset)):
    #for i in range(10):
        try:
            inputs = [dataset[i]]

            gt_pth=inputs[0][1]["file_name"].replace('/img/','/manually annotated masks/').replace('.jpg','.png')
            
            name=inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')
            pre_fig_root='/home/zyf/code/Saliency-Ranking-main/tools/preds_RVSOD_model_0009999/preds_RVSOD_model_0009999/'

            pre_fig=pre_fig_root+inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')
            our_pth='/home/zyf/code/Saliency-Ranking-main/tools/draw_mask_85/'+inputs[0][1]["file_name"].split('/')[-1].replace('.jpg','.png')
            #gt_pth=inputs[0][1]["file_name"].replace('/img/','/manually annotated masks/').replace('.jpg','.png')
            gt_pth='/home/zyf/code/Saliency-Ranking-main/tools/draw_gt/'+inputs[0][1]["file_name"].split('/')[-1]

            image = Image.open(pre_fig)
            image_gt=Image.open(gt_pth)
            image_our=Image.open(our_pth)
            size = image.size
            
            if size!=image_gt.size:
                image_gt=image_gt.resize(size,Image.NEAREST)
            if size!=image_our.size:
                image_our=image_our.resize(size,Image.NEAREST)

            maskdata_gt=np.array(image_gt.convert('L'))
            maskdata_gt[maskdata_gt !=0] = 1
            

            maskdata_our=np.array(image_our.convert('L'))
            maskdata_our[maskdata_our !=0] = 1

            
            maskdata=np.array(image.convert('L'))
            maskdata[maskdata !=0] = 1

            maskdata=maskdata.reshape(-1)
            maskdata_gt=maskdata_gt.reshape(-1)
            maskdata_our=maskdata_our.reshape(-1)

            
            if len(maskdata_gt)==len(maskdata_our):
                mae=0
                for k in range(len(maskdata_our)):
                    if maskdata_gt[k]!=maskdata_our[k]:
                        mae+=1
                mae=mae/len(maskdata_our)
                #print(mae)
                mae_sum+=mae
            else:
                no_figure+=1
                continue

            print('\r{}/{}'.format(i, len(dataset)), end="", flush=True)
            num+=1

        except:
            print(name)
            no_figure+=1

    print('mae:',mae_sum/len(dataset))
    print(no_figure)

    return 0

def cal_RVSOD():
    root_gt='/home/zyf/code/Saliency-Ranking-main/tools/preds_RVSOD_model/gt_test/'
    root_data='/home/zyf/code/Saliency-Ranking-main/tools/preds_RVSOD_model/preds_RVSOD_model_0009999_resize_2/'
    gt=os.listdir(root_gt)
    data=os.listdir(root_data)
    mae_sum=0
    num=0
    for i in range(len(data)):
        img_gt_pth=root_gt+gt[i]
        img_data_pth=root_data+data[i]
        img_gt=Image.open(img_gt_pth)
        img_data=Image.open(img_data_pth)

        if img_data.size!=img_gt.size:
                img_data=img_data.resize(img_gt.size,Image.NEAREST)

        maskdata_gt=np.array(img_gt.convert('L'))
        my_gt=np.zeros([len(maskdata_gt), len(maskdata_gt[0])])
        my_gt[maskdata_gt!=0]=1
        # for j in range(len(maskdata_gt)):
        #     for k in range(len(maskdata_gt[j])):
        #         if maskdata_gt[j][k]!=0:
        #             maskdata_gt[j][k]=1
        
        maskdata_data=np.array(img_data.convert('L'))
        my_data=np.zeros([len(maskdata_data), len(maskdata_data[0])])
        my_data[maskdata_data!=0]=1
        # for j in range(len(maskdata_data)):
        #     for k in range(len(maskdata_data[j])):
        #         if maskdata_data[j][k]!=0:
        #             maskdata_data[j][k]=1

        # maskdata_data=maskdata_data.reshape(-1)
        # maskdata_gt=maskdata_gt.reshape(-1)
        maskdata_data=my_data.reshape(-1)
        maskdata_gt=my_gt.reshape(-1)
            
        if len(maskdata_gt)==len(maskdata_data):
            mae=0
            for k in range(len(maskdata_gt)):
                if maskdata_gt[k]!=maskdata_data[k]:
                    mae+=1
            mae=mae/len(maskdata_data)
            mae_sum+=mae
            num+=1
        else:
            no_figure+=1
            print(1)
            continue

        print('\r{}/{}'.format(i, len(data)), end="", flush=True)
    print()
    print('MAE:',mae_sum/num)
    return 0

def get_gt():
    imgfile=['/home/zyf/code/Saliency-Ranking-main/dataset/RVSOD/RVSOD/test/ranking saliency masks/img']
    for pth in imgfile:
        videoL=os.listdir(pth)
        for i in range(len(videoL)):
            imageName=os.listdir(pth+'/'+videoL[i])
            imageName.sort(key=lambda l: int(l.split('_')[-1].split('.')[0]))
            for j in range(len(imageName)):
                cur=pth+'/'+videoL[i]+'/'+imageName[j]
                png=Image.open(cur)
                png.save('/home/zyf/code/Saliency-Ranking-main/tools/preds_RVSOD_model_0009999/gt_test/'+imageName[j])


    return 0
if __name__ == '__main__':
    #print(np.absolute([-1,1,0]))
    inference()
    #eval_mae()
    #resize()
    #get_gt()
    #cal_RVSOD()
    pass