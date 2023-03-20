import numpy as np
import cv2
import os
import random
import tensorflow as tf
from Network import Network2
import argparse


def get_patches(base_path, img_name, no_patches):
    train_img_path = os.path.join(base_path, img_name)
    print(train_img_path)
    train_img = cv2.imread(train_img_path)

    # Down-scaling and converting the image to grayscale
    down_width = 320
    down_height = 240
    down_points = (down_width, down_height)
    scld_img = cv2.resize(train_img, down_points, interpolation=cv2.INTER_AREA)
    scld_img = cv2.cvtColor(scld_img, cv2.COLOR_BGR2GRAY)

    random_patches = list()
    homographies = list()
    for num in range(no_patches):
        x_range = list(range(scld_img.shape[1]))
        y_range = list(range(scld_img.shape[0]))
        patch_size = 128

        x1 = np.random.choice(x_range)
        x2 = x1 + patch_size
        y1 = np.random.choice(y_range)
        y2 = y1 + patch_size
        threshold = 10
        prtrbnce_rng = np.arange(-32, 32)
        while True:
            if (threshold < x1 - max(prtrbnce_rng)) and (x2 + max(prtrbnce_rng) < scld_img.shape[1] - threshold) and \
                    (threshold < y1 - max(prtrbnce_rng)) and (y2 + max(prtrbnce_rng) < scld_img.shape[0] - threshold):
                # print("Choice of patch points: ", (x1, y1), (x2, y1), (x2, y2), (x1, y2))
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
        pcp1 = (cp1[0] + random.choice(prtrbnce_rng), cp1[1] + random.choice(prtrbnce_rng))
        pcp2 = (cp2[0] + random.choice(prtrbnce_rng), cp2[1] + random.choice(prtrbnce_rng))
        pcp3 = (cp3[0] + random.choice(prtrbnce_rng), cp3[1] + random.choice(prtrbnce_rng))
        pcp4 = (cp4[0] + random.choice(prtrbnce_rng), cp4[1] + random.choice(prtrbnce_rng))

        # Obtaining the Homography matrix
        patch_corners = np.float32([cp1, cp2, cp3, cp4])
        prtrbd_crnrs = np.float32([pcp1, pcp2, pcp3, pcp4])
        H_ab = cv2.getPerspectiveTransform(prtrbd_crnrs, patch_corners)  # Homography matrix
        H_ba = np.linalg.inv(H_ab)
        warped_image = cv2.warpPerspective(scld_img, H_ba, (scld_img.shape[1], scld_img.shape[0]))

        # Gathering the patches from the images
        patch_A = scld_img[y1:y2, x1:x2]
        patch_B = warped_image[y1:y2, x1:x2]
        patches = np.dstack((patch_A, patch_B))

        # Computing the difference between the original and perturbed points
        diff1 = tuple(map(lambda i, j: i - j, pcp1, cp1))
        diff2 = tuple(map(lambda i, j: i - j, pcp2, cp2))
        diff3 = tuple(map(lambda i, j: i - j, pcp3, cp3))
        diff4 = tuple(map(lambda i, j: i - j, pcp4, cp4))
        H4Pt = np.array([diff1, diff2, diff3, diff4])

        random_patches.append(patches)
        homographies.append(H4Pt)
    return random_patches, homographies


def TrainOperation(BasePath, MiniBatchSize, CheckPointPath, NumEpochs):
    Img_Dir = BasePath
    file_names = [str(i) + '.jpg' for i in range(1, 5000)]
    all_images = list()
    actual_outputs = list()
    for f_name in file_names:
        my_patches, my_H4Pt = get_patches(Img_Dir, f_name, 1)
        for i in range(len(my_patches)):
            x_train = my_patches[i].astype('float32')
            y_train = my_H4Pt[i].flatten().astype('float32')
            all_images.append(x_train)
            actual_outputs.append(y_train)
    my_model = Network2.get_homography_model()
    output = my_model.fit(x=np.array(all_images), y=np.array(actual_outputs), epochs=NumEpochs, batch_size=MiniBatchSize, shuffle=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckPointPath, save_weights_only=False,
                                                                   monitor='loss', save_freq='epoch', save_best_only=False)


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/naveen/CMSC733/nmangla_p1/Phase2/Data/Train',
                        help='Base path of images')
    Parser.add_argument('--CheckPointPath', default='/home/naveen/CMSC733/nmangla_p1/Phase2/Checkpoints/',
                        help='Path to save Checkpoints, Default: ../Checkpoints/')
    # Parser.add_argument('--ModelType', default='Unsup',
    #                     help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=25, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:128')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0,
                        help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    MiniBatchSize = Args.MiniBatchSize
    CheckPointPath = Args.CheckPointPath
    TrainOperation(BasePath, MiniBatchSize, CheckPointPath, NumEpochs)


if __name__ == '__main__':
    main()



