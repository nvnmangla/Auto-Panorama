

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s):
Chahat Deep Singh (chahat@terpmail.umd.edu)
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

from cProfile import label
from email.parser import Parser
from Network.Network import SupHomographyModel
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import copy
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Add any python libraries here



def plot_corners(image, corner_pts, color):
    for crnr in corner_pts:
        crnr = (int(crnr[0]), int(crnr[1]))
        centr_coord = crnr
        radius = 1
        thickness = -1
        patch_img = cv2.circle(image, centr_coord, radius, color, thickness)
    cv2.imshow('Patch', patch_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def generate_patches(BasePath):
    # RIndx = random.randint(1, 5000)
    H14 = []
    for RIndx in range(1, 5000):
        img_name = BasePath + str(RIndx) + '.jpg'
        train_img = cv2.imread(img_name)
        gs_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        down_width = 320
        down_height = 240
        down_points = (down_width, down_height)

        scld_img = cv2.resize(
            gs_img, down_points, interpolation=cv2.INTER_AREA)

        p = [i for i in range(-32, 32)]
        max_p = 32

        x_range = list(range(scld_img.shape[1]))
        y_range = list(range(scld_img.shape[0]))
        patch_size = 128
        for j in range(10):
            x1 = np.random.choice(x_range)
            x2 = x1 + patch_size
            y1 = np.random.choice(y_range)
            y2 = y1 + patch_size
            threshold = 10

            while True:
                # print(x1, y1)
                if (x2 + max_p < scld_img.shape[1] - threshold) and (x1 - max_p > threshold) and (
                        y2 + max_p < scld_img.shape[0] - threshold) and (y1 - max_p > threshold):
                    print("Choice of patch points: ", (x1, y1),
                        (x2, y1), (x2, y2), (x1, y2))
                    break
                else:
                    x1 = np.random.choice(x_range)
                    y1 = np.random.choice(y_range)
                    x2 = x1 + patch_size
                    y2 = y1 + patch_size

            # Corner points of the patch
            cp1 = (x1, y1)
            cp2 = (x2, y1)
            cp3 = (x1, y2)
            cp4 = (x2, y2)
            # Peturbed corners of the patch
            pcp1 = (cp1[0] + random.choice(p), cp1[1] + random.choice(p))
            pcp2 = (cp2[0] + random.choice(p), cp2[1] + random.choice(p))
            pcp3 = (cp3[0] + random.choice(p), cp3[1] + random.choice(p))
            pcp4 = (cp4[0] + random.choice(p), cp4[1] + random.choice(p))

            patch_corners = np.float32([cp1, cp2, cp3, cp4])

            prtrbd_crnrs = np.float32([pcp1, pcp2, pcp3, pcp4])
            H_ab = cv2.getPerspectiveTransform(
                prtrbd_crnrs, patch_corners)  # Homography matrix
            H_ba = np.linalg.inv(H_ab)
            warped_image = cv2.warpPerspective(
                scld_img, H_ba, (scld_img.shape[1], scld_img.shape[0]))
            patch_A = scld_img[y1:y2, x1:x2]
            # print("patchA=",patch_A.shape)
            patch_B = warped_image[y1:y2, x1:x2]
            

            in_image = np.dstack((patch_A, patch_B))
            np.save(BasePath+'patches/'+str(RIndx)+"_"+str(j), in_image)
            h_14 = prtrbd_crnrs-patch_corners
            H14.append(h_14)
    
    with open(BasePath+'labels.txt', 'w') as f:
        for h14 in H14:    
            for i in range(h14.shape[0]):
                for j in range(h14.shape[1]):
                    print("elem = ",h14[i,j])
                    f.write(str(h14[i,j]))
            f.write('\n')

    return in_image, h_14, prtrbd_crnrs, patch_corners, train_img


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/naveen/CMSC733/nmangla_p1/Phase2/Data/Train/',
                        help='Give your path')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/naveen/CMSC733/nmangla_p1/Phase2/Data/',
                        help='Path to load latest model from, Default:ModelPath')

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    ModelPath = Args.ModelPath

    ImgPH = tf.placeholder(tf.float32, shape=(1, 128, 128, 2))
    PatchSize = 128
    p,_,_,_,_ = generate_patches(BasePath)
    


if __name__ == '__main__':
    main()
