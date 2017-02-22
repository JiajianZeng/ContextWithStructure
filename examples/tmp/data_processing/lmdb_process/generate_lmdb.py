import os
import time
import math
import sys
import lmdb
from os.path import join, exists
import cv2
import numpy as np
import h5py
import caffe
from common import shuffle_in_unison_scary, logger, createDir, processImage
from common import getDataFromTxt
from utils import show_landmark, flip, rotate


TRAIN = 'dataset/train'

def generate_lmdb(ftxt, output, fname, isColor, img_size, argument=False):

    data = getDataFromTxt(ftxt)
    F_imgs = []
    F_landmarks = []
    F_eyedist = []
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
            face_flipped = cv2.resize(face_flipped, (img_size, img_size))
            F_imgs.append(face_flipped.reshape((channel, img_size, img_size)))
            F_landmarks.append(landmark_flipped.reshape(10))
            ### rotation
            if np.random.rand() > 0.5:
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, \
                    bbox.reprojectLandmark(landmarkGt), 5, channel)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size, img_size))
                F_imgs.append(face_rotated_by_alpha.reshape((channel, img_size, img_size)))
                F_landmarks.append(landmark_rotated.reshape(10))
                ### flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped = cv2.resize(face_flipped, (img_size, img_size))
                F_imgs.append(face_flipped.reshape((channel, img_size, img_size)))
                F_landmarks.append(landmark_flipped.reshape(10))
            ### rotation
            if np.random.rand() > 0.5:
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, \
                    bbox.reprojectLandmark(landmarkGt), -5, channel)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size, img_size))
                F_imgs.append(face_rotated_by_alpha.reshape((channel, img_size, img_size)))
                F_landmarks.append(landmark_rotated.reshape(10))
                ### flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped = cv2.resize(face_flipped, (img_size, img_size))
                F_imgs.append(face_flipped.reshape((channel, img_size, img_size)))
                F_landmarks.append(landmark_flipped.reshape(10))

        f_face = cv2.resize(f_face, (img_size, img_size))

        f_face = f_face.reshape((channel, img_size, img_size))
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
 
    #F_imgs = processImage(F_imgs,channel)
    shuffle_in_unison_scary(F_imgs, F_landmarks, F_eyedist)
    # full face
    base = join(OUTPUT, '1_F')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    
    in_db = lmdb.open(output + '_data', map_size=1e12)  
    with in_db.begin(write=True) as in_txn :  
        for in_idx,in_ in enumerate(F_imgs) :  
            im = in_;  
            im = im[::-1,:,:]  
            #im = im.transpose((2, 0, 1))  
            im_dat = caffe.io.array_to_datum(im)  
            #in_txn.put(in_idx.encode('ascii'), im_dat.SerializeToString())  
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())  
    in_db.close() 
    if(argument):
        cmd = "../../build/tools/compute_image_mean " + output + "_data " + base + "/train_mean.binaryproto"
    else:
        cmd = "../../build/tools/compute_image_mean " + output + "_data " + base + "/test_mean.binaryproto"
    os.system(cmd)

    in_label = lmdb.open(output + '_landmark', map_size=1e12)  
    counter_label = 0  
    with in_label.begin(write=True) as in_txn :  
        for landmark in F_landmarks:  
            datum = caffe.proto.caffe_pb2.Datum()  
            datum.channels = landmark.shape[0]
            datum.height = 1
            datum.width = 1
            datum.float_data.extend(landmark.astype(float).flat)
            in_txn.put("{:0>10d}".format(counter_label), datum.SerializeToString())  
            counter_label += 1  
    in_label.close()  

    in_eyedist = lmdb.open(output + '_eyedist', map_size=1e12)  
    counter_eyedist = 0  
    with in_eyedist.begin(write=True) as in_txn :  
        for eyedist in F_eyedist:  
            datum = caffe.proto.caffe_pb2.Datum()  
            datum.channels = eyedist.shape[0]
            datum.height = 1
            datum.width = 1
            datum.float_data.extend(eyedist.astype(float).flat)
            in_txn.put("{:0>10d}".format(counter_eyedist), datum.SerializeToString())  
            counter_eyedist += 1  
    in_eyedist.close()  


if __name__ == '__main__':
    # train data
    assert(len(sys.argv) == 4)
    isColor = int(sys.argv[1])
    img_size = int(sys.argv[2])
    OUTPUT = sys.argv[3]
    if not exists(OUTPUT): os.mkdir(OUTPUT)
    assert(exists(TRAIN) and exists(OUTPUT))
    train_txt = join(TRAIN, 'trainImageList.txt')
    generate_lmdb(train_txt, OUTPUT, 'train_lmdb', isColor, img_size, argument=True)

    test_txt = join(TRAIN, 'testImageList.txt')
    generate_lmdb(test_txt, OUTPUT, 'test_lmdb', isColor, img_size)

  
    # Done
