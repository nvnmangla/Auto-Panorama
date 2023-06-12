import glob
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms
import os
# from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import matplotlib.pyplot as plt
import glob
cv_img = []

rho = 32
patch_size = 128



for i in glob.glob("Data/Train/Train/*.jpg"):

    img = cv2.imread(i)
    img=cv2.resize(img,(320,240))
    
    
    x,y,_ = img.shape

    
    p_y = np.random.randint(rho,x-rho-patch_size)
    p_x = np.random.randint(rho,y-rho-patch_size)
    
    
    src_pts = np.array([[p_x,p_y],[p_x+patch_size,p_y],[p_x+patch_size,p_y+patch_size],[p_x,p_y+patch_size]],dtype=np.float32)

    dst_pts = []
    for i in range(4):
        p_x_  = np.random.randint(-rho,rho)    
        p_y_  = np.random.randint(-rho,rho) 

        p1 = src_pts[i][0] + p_x_
        p2 = src_pts[i][1] + p_y_
        
        dst_pts.append([p1,p2])
    dst_pts = np.array(dst_pts,dtype=np.float32)
    H_AB = cv2.getPerspectiveTransform(src_pts,dst_pts)
    H_BA = np.linalg.inv(H_AB)

    img2 = cv2.warpPerspective(img,H_AB,(y,x))
    
    
    patchA = img[p_x:p_x+patch_size, p_y:p_y+patch_size]
    patchB = img2[p_x:p_x+patch_size, p_y:p_y+patch_size]
    
    
    cv2.imshow("window",patchA)
    cv2.imshow("windo1w",patchB)
    
    cv2.waitKey(0)
    
    
    break

cv2.destroyAllWindows()