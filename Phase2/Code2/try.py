#!/usr/bin/python3
import cv2
import numpy as np
import random
import math
import os

def GeneratePatches(path,patch_size=128,rho=32):

    # idx=random.randint(0,len(image_names)-1)
    # # print(idx)
    # path=ImagesPath + os.sep + image_names[idx]
    # print(path)
    
    img=cv2.imread(path,0)  
    # img=cv2.resize(img,(320,240))
    
    M,N=img.shape

    if((M<patch_size+2*rho+1) | (N<patch_size+2*rho+1)):
        return False,_,_,_,_,_

    # Top left corner of patch selected
    # print("M rand is between range: ",rho," to ",M-patch_size-rho) 
    # print("N rand is between range: ",rho," to ",N-patch_size-rho)
    if(M<229 or N <229):
        return False,_,_,_,_,_

    Ca1=np.array([random.randint(rho,M-patch_size-rho),random.randint(rho,N-patch_size-rho)])

    Ca=np.array([[Ca1[1],Ca1[0]],
                 [Ca1[1]+patch_size,Ca1[0]],
                 [Ca1[1]+patch_size,Ca1[0]+patch_size ],
                 [Ca1[1],Ca1[0]+patch_size]])

    # Finding patch after random perturbations
    Cb=np.zeros(Ca.shape,dtype=np.int)

    for i in range(4):
        rx=random.randint(-rho,rho)
        ry=random.randint(-rho,rho)
        Cb[i,0]=Ca[i,0] + ry
        Cb[i,1]=Ca[i,1] + rx

    # Visualize patches and perspective transform:
    # cv2.polylines(img,[Ca],True,(0,0,255),2)
    # cv2.polylines(img,[Cb],True,(0,255,0),2)
    # cv2.imshow('patch selection',img)
    # cv2.waitKey(0)


    # The following two methods of finding inverse perspective transform are equivalent
    # Method 1
    # H=cv2.getPerspectiveTransform(Cb.astype(np.float32),Ca.astype(np.float32))
    # Method 2
    H1=cv2.getPerspectiveTransform(Ca.astype(np.float32),Cb.astype(np.float32))
    H=np.linalg.inv(H1)
    # print(img.shape)

    # print(img.shape[:-1])
    h,w=img.shape
    warped = cv2.warpPerspective(img,H,(w,h))

    patch_A = img[Ca[0,1]:Ca[3,1],Ca[0,0]:Ca[1,0]]
    patch_B = warped[Ca[0,1]:Ca[3,1],Ca[0,0]:Ca[1,0]]


    # See patches 
    # cv2.imshow('patchA',patch_A)
    # cv2.imshow('patchB',patch_B)
    # cv2.waitKey()

    H_4pt=Cb-Ca
    # check
    # mat=cv2.getPerspectiveTransform((H_4pt+Ca).astype(np.float32),Ca.astype(np.float32))
    # warpcheck = cv2.warpPerspective(img,mat,(w,h))
    # cv2.imshow('original',img)
    # cv2.imshow('warpcheck',warpcheck)
    # cv2.imshow('warped',warped)
    # cv2.waitKey(0)
    # ------
    # print(H_4pt.shape)
    # print(patch_B.shape)
    if(H_4pt.shape[0]!=4):
        return False,_, _, _,_,_

    return True, patch_A, patch_B, H_4pt,Ca,Cb,img,warped,img


_,a,b,_,_,_,_,c,d = GeneratePatches("Data/Train/Train/1.jpg",128,32)

cv2.imshow("window1",c)

cv2.imshow("window",d)

cv2.waitKey(0)

    

# if __name__=="__main__":
#     ImagesPath=r"./../Data/Train"
#     PatchAPath=r"./../Data/testing/PatchA"
#     PatchBPath=r"./../Data/testing/PatchB"
#     H4Path=r"./../Data/testing/H4"

#     if not os.path.exists(ImagesPath):
#         print("The images path does not exist!")
#         exit
#     if not os.path.exists(PatchAPath):
#         os.makedirs(PatchAPath)

#     if not os.path.exists(PatchBPath):
#         os.makedirs(PatchBPath)

#     if not os.path.exists(H4Path):
#         os.makedirs(H4Path)

#     image_names=[]
#     for dir,subdir,files in os.walk(ImagesPath):
#         for file in files:
#             image_names.append(file)
#     count=0
#     n_samples=10
#     H_4pt_array=[]
#     Ca_array=[]
#     Cb_array=[]

#     while(count<n_samples):
#         retval, patch_A, patch_B, H_4pt, Ca,Cb=GeneratePatches(ImagesPath,image_names)
#         # cv2.imshow('patch_A',patch_A)
#         # cv2.imshow('patch_B',patch_B)
#         # cv2.waitKey()
        
#         if(retval):
#             cv2.imwrite(PatchAPath+os.sep+str(count)+'.jpg',patch_A)
#             cv2.imwrite(PatchBPath+os.sep+str(count)+'.jpg',patch_B)
#             H_4pt_array.append(H_4pt)
#             Ca_array.append(Ca)
#             Cb_array.append(Cb)
#             count+=1
#         else:
#             print("Exception found in an image!")
#     H_4pt_array=np.array(H_4pt_array)
#     Ca_array=np.array(Ca_array)

#     Cb_array=np.array(Cb_array)

#     np.save(H4Path+os.sep+"H4.npy",H_4pt_array)
#     np.save(H4Path+os.sep+"Ca.npy",Ca_array)

#     np.save(H4Path+os.sep+"Cb.npy",Cb_array)