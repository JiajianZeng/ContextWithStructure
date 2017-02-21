import os
import time
import math
import sys
from os.path import join, exists
import cv2
import numpy as np
import h5py
from common import shuffle_in_unison_scary_color, logger, createDir, processImage
from common import getDataFromTxt
from utils import show_landmark, flip, rotate


TRAIN = 'dataset/train'

def generate_hdf5(ftxt, output, fname, isColor, img_size1, img_size2, argument=False):

    data = getDataFromTxt(ftxt)
    F_imgs_pair1 = []
    F_landmarks_pair1 = []
    F_eyedist_pair1 = []
    F_imgs_pair2 = []
    F_landmarks_pair2 = []
    F_eyedist_pair2 = []
    for (imgPath, bbox, landmarkGt, eyeDist) in data:
        if(isColor):
           channel = 3
           img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_COLOR)
        else:
           channel = 1
           img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        # F
        f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
        f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]

        ## data argument
        if argument and np.random.rand() > -1:
            ### flip
            face_flipped, landmark_flipped = flip(f_face, landmarkGt)
            face_flipped_pair1 = cv2.resize(face_flipped, (img_size1, img_size1))
            face_flipped_pair2 = cv2.resize(face_flipped, (img_size2, img_size2))
            F_imgs_pair1.append(face_flipped_pair1.reshape((channel, img_size1, img_size1)))
            F_imgs_pair2.append(face_flipped_pair2.reshape((channel, img_size2, img_size2)))
            F_landmarks_pair1.append(landmark_flipped.reshape(10))
            F_landmarks_pair2.append(landmark_flipped.reshape(10))
            ### rotation
            if np.random.rand() > 0.5:
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, \
                    bbox.reprojectLandmark(landmarkGt), 5, channel)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha_pair1 = cv2.resize(face_rotated_by_alpha, (img_size1, img_size1))
                face_rotated_by_alpha_pair2 = cv2.resize(face_rotated_by_alpha, (img_size2, img_size2))
                F_imgs_pair1.append(face_rotated_by_alpha_pair1.reshape((channel, img_size1, img_size1)))
                F_imgs_pair2.append(face_rotated_by_alpha_pair2.reshape((channel, img_size2, img_size2)))
                F_landmarks_pair1.append(landmark_rotated.reshape(10))
                F_landmarks_pair2.append(landmark_rotated.reshape(10))
                ### flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped_pair1 = cv2.resize(face_flipped, (img_size1, img_size1))
                face_flipped_pair2 = cv2.resize(face_flipped, (img_size2, img_size2))
                F_imgs_pair1.append(face_flipped_pair1.reshape((channel, img_size1, img_size1)))
                F_imgs_pair2.append(face_flipped_pair2.reshape((channel, img_size2, img_size2)))
                F_landmarks_pair1.append(landmark_flipped.reshape(10))
                F_landmarks_pair2.append(landmark_flipped.reshape(10))
            ### rotation
            if np.random.rand() > 0.5:
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, \
                    bbox.reprojectLandmark(landmarkGt), -5, channel)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha_pair1 = cv2.resize(face_rotated_by_alpha, (img_size1, img_size1))
                face_rotated_by_alpha_pair2 = cv2.resize(face_rotated_by_alpha, (img_size2, img_size2))
                F_imgs_pair1.append(face_rotated_by_alpha_pair1.reshape((channel, img_size1, img_size1)))
                F_imgs_pair2.append(face_rotated_by_alpha_pair2.reshape((channel, img_size2, img_size2)))
                F_landmarks_pair1.append(landmark_rotated.reshape(10))
                F_landmarks_pair2.append(landmark_rotated.reshape(10))
                ### flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped_pair1 = cv2.resize(face_flipped, (img_size1, img_size1))
                face_flipped_pair2 = cv2.resize(face_flipped, (img_size2, img_size2))
                F_imgs_pair1.append(face_flipped_pair1.reshape((channel, img_size1, img_size1)))
                F_imgs_pair2.append(face_flipped_pair2.reshape((channel, img_size2, img_size2)))
                F_landmarks_pair1.append(landmark_flipped.reshape(10))
                F_landmarks_pair2.append(landmark_flipped.reshape(10))

        f_face_pair1 = cv2.resize(f_face, (img_size1, img_size1))
        f_face_pair2 = cv2.resize(f_face, (img_size2, img_size2))
        f_face_pair1 = f_face_pair1.reshape((channel, img_size1, img_size1))
        f_face_pair2 = f_face_pair2.reshape((channel, img_size2, img_size2))
        f_landmark = landmarkGt.reshape((10))
        f_eyeDist = eyeDist.reshape((1))
        F_imgs_pair1.append(f_face_pair1)
        F_imgs_pair2.append(f_face_pair2)
        F_landmarks_pair1.append(f_landmark)
        F_eyedist_pair1.append(f_eyeDist)
        F_landmarks_pair2.append(f_landmark)
        F_eyedist_pair2.append(f_eyeDist)
        # EN
        # en_bbox = bbox.subBBox(-0.05, 1.05, -0.04, 0.84)
        # en_face = img[en_bbox.top:en_bbox.bottom+1,en_bbox.left:en_bbox.right+1]

        ## data argument
        

    #imgs, landmarks = process_images(ftxt, output)

    F_imgs_pair1, F_landmarks_pair1, F_eyedist_pair1 = np.asarray(F_imgs_pair1), np.asarray(F_landmarks_pair1), np.asarray(F_eyedist_pair1)
    F_imgs_pair2, F_landmarks_pair2, F_eyedist_pair2 = np.asarray(F_imgs_pair2), np.asarray(F_landmarks_pair2), np.asarray(F_eyedist_pair2)

    F_imgs_pair1 = processImage(F_imgs_pair1,channel)
    F_imgs_pair2 = processImage(F_imgs_pair2,channel)
    #shuffle_in_unison_scary(F_imgs, F_landmarks, F_eyedist)
    shuffle_in_unison_scary_color(F_imgs_pair1, F_imgs_pair2, F_landmarks_pair1, F_landmarks_pair2, F_eyedist_pair1, F_eyedist_pair2)
    # full face
    base = join(OUTPUT, 'pair1')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = F_imgs_pair1.astype(np.float32)
        h5['landmark'] = F_landmarks_pair1.astype(np.float32)
        h5['eyedist'] = F_eyedist_pair1.astype(np.float32)

    base = join(OUTPUT, 'pair2')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = F_imgs_pair2.astype(np.float32)
        h5['landmark'] = F_landmarks_pair2.astype(np.float32)
        h5['eyedist'] = F_eyedist_pair2.astype(np.float32)

if __name__ == '__main__':
    # train data
    assert(len(sys.argv) == 5)
    isColor = int(sys.argv[1])
    img_size1 = int(sys.argv[2])
    img_size2 = int(sys.argv[3])
    OUTPUT = sys.argv[4]
    if not exists(OUTPUT): os.mkdir(OUTPUT)
    assert(exists(TRAIN) and exists(OUTPUT))
    train_txt = join(TRAIN, 'trainImageList.txt')
    generate_hdf5(train_txt, OUTPUT, 'train.h5', isColor, img_size1, img_size2, argument=True)

    test_txt = join(TRAIN, 'testImageList.txt')
    generate_hdf5(test_txt, OUTPUT, 'test.h5', isColor, img_size1, img_size2)

    with open(join(OUTPUT, 'pair1/train.txt'), 'w') as fd:
        fd.write(OUTPUT+'/pair1/train.h5')
    with open(join(OUTPUT, 'pair1/test.txt'), 'w') as fd:
        fd.write(OUTPUT+'/pair1/test.h5')
    with open(join(OUTPUT, 'pair2/train.txt'), 'w') as fd:
        fd.write(OUTPUT+'/pair2/train.h5')
    with open(join(OUTPUT, 'pair2/test.txt'), 'w') as fd:
        fd.write(OUTPUT+'/pair2/test.h5')
  
    # Done
