import os
import time
import math
from os.path import join, exists
import cv2
import numpy as np
import h5py
from common import shuffle_in_unison_scary, logger, createDir, processImage
from common import getDataFromTxt
from utils import show_landmark, flip, rotate


TRAIN = 'dataset/train'
OUTPUT = 'train'
if not exists(OUTPUT): os.mkdir(OUTPUT)
assert(exists(TRAIN) and exists(OUTPUT))


def generate_hdf5(ftxt, output, fname, argument=False):

    data = getDataFromTxt(ftxt)
    F_imgs = []
    F_landmarks = []
    F_eyedist = []

    for (imgPath, bbox, landmarkGt, eyeDist) in data:
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
            face_flipped = cv2.resize(face_flipped, (227, 227))
            F_imgs.append(face_flipped.reshape((1, 227, 227)))
            F_landmarks.append(landmark_flipped.reshape(10))
            ### rotation
            if np.random.rand() > 0.5:
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, \
                    bbox.reprojectLandmark(landmarkGt), 5)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (227, 227))
                F_imgs.append(face_rotated_by_alpha.reshape((1, 227, 227)))
                F_landmarks.append(landmark_rotated.reshape(10))
                ### flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped = cv2.resize(face_flipped, (227, 227))
                F_imgs.append(face_flipped.reshape((1, 227, 227)))
                F_landmarks.append(landmark_flipped.reshape(10))
            ### rotation
            if np.random.rand() > 0.5:
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, \
                    bbox.reprojectLandmark(landmarkGt), -5)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (227, 227))
                F_imgs.append(face_rotated_by_alpha.reshape((1, 227, 227)))
                F_landmarks.append(landmark_rotated.reshape(10))
                ### flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped = cv2.resize(face_flipped, (227, 227))
                F_imgs.append(face_flipped.reshape((1, 227, 227)))
                F_landmarks.append(landmark_flipped.reshape(10))

        f_face = cv2.resize(f_face, (227, 227))

        f_face = f_face.reshape((1, 227, 227))
        f_landmark = landmarkGt.reshape((10))
        f_eyeDist = eyeDist.reshape((1))
        F_imgs.append(f_face)
        F_landmarks.append(f_landmark)
        F_eyedist.append(f_eyeDist)
        # EN
        # en_bbox = bbox.subBBox(-0.05, 1.05, -0.04, 0.84)
        # en_face = img[en_bbox.top:en_bbox.bottom+1,en_bbox.left:en_bbox.right+1]

        ## data argument
        

    #imgs, landmarks = process_images(ftxt, output)

    F_imgs, F_landmarks, F_eyedist = np.asarray(F_imgs), np.asarray(F_landmarks), np.asarray(F_eyedist)
 
    F_imgs = processImage(F_imgs)
    shuffle_in_unison_scary(F_imgs, F_landmarks, F_eyedist)
    # full face
    base = join(OUTPUT, '1_F')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = F_imgs.astype(np.float32)
        h5['landmark'] = F_landmarks.astype(np.float32)
        h5['eyedist'] = F_eyedist.astype(np.float32)

if __name__ == '__main__':
    # train data
    train_txt = join(TRAIN, 'trainImageList.txt')
    generate_hdf5(train_txt, OUTPUT, 'train.h5', argument=True)

    test_txt = join(TRAIN, 'testImageList.txt')
    generate_hdf5(test_txt, OUTPUT, 'test.h5')

    with open(join(OUTPUT, '1_F/train.txt'), 'w') as fd:
        fd.write('train/1_F/train.h5')
    with open(join(OUTPUT, '1_F/test.txt'), 'w') as fd:
        fd.write('train/1_F/test.h5')
  
    # Done
