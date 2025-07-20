import time
import os
import torch
from tqdm import tqdm
import numpy as np
import copy
import cv2

a='./draw_box'
b='./draw_mask'

filesa=os.listdir(a)
filesb=os.listdir(b)

imgsa=[]
imgsb=[]
for i in range(min(len(filesb),len(filesa))):
    
    imgsa.append(a+'/'+filesa[i])
    imgsb.append(b+'/'+filesb[i])
    
print(len(imgsa))
print(len(imgsb))
for i in range(len(imgsa)):
    for j in range(len(imgsb)):
        
        if imgsa[i].split('/')[-1].split('.')[0]==imgsb[j].split('/')[-1].split('.')[0]:

            img1=cv2.imread(imgsa[i])
            img2=cv2.imread(imgsb[j])
    
            imgd=np.vstack([img1,img2])

            cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/draw_TS/'+imgsa[i].split('/')[-1], imgd)
            
            break
        
        else:
            continue

# a='./draw_res_epoch1'
# b='./draw_res_epoch9'

# filesa=os.listdir(a)
# filesb=os.listdir(b)

# imgsa=[]
# imgsb=[]
# for i in range(min(len(filesb),len(filesa))):
    
#     imgsa.append(a+'/'+filesa[i])
#     imgsb.append(b+'/'+filesb[i])
    

# for i in range(len(imgsa)):
#     for j in range(len(imgsb)):
        
#         if imgsa[i].split('/')[-1]==imgsb[j].split('/')[-1]:

#             img1=cv2.imread(imgsa[i])
#             img2=cv2.imread(imgsb[j])

#             imgd=np.hstack([img1,img2])
    
#             cv2.imwrite('/home/zyf/code/Saliency-Ranking-main/tools/draw_res_compare/'+imgsa[i].split('/')[-1], imgd)
            
#             break
        
#         else:
#             continue