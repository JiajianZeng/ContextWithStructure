import os, sys
import time
from functools import partial
import cv2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from common import getCNNs
from common import getDataFromTxt, logger
from common import shuffle_in_unison_scary, createDir, processImage


TXT = 'dataset/train/testImageList.txt'
template = '''################## Summary #####################
Test Number: %d
Time Consume: %.03f s
FPS: %.03f
LEVEL - %d
Mean Error:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
Failure:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
'''

def evaluateError(landmarkGt, landmarkP, bbox):
    e = np.zeros(5)
    for i in range(5):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    e = e / bbox.w
    print 'landmarkGt'
    print landmarkGt
    print 'landmarkP'
    print landmarkP
    print 'error', e
    return e

def getResult(img,bbox):
    # F
    img_size = 227
    F = getCNNs(level = 1)[0]

    f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]


    f_face = cv2.resize(f_face, (img_size, img_size))

    f_face = f_face.reshape((1, 3, img_size, img_size))
    f_face = processImage(f_face)
    f = F.forward(f_face)
  
    return f
     
def E():
    F_imgs = []
    img_size = 227
    data = getDataFromTxt(TXT)
    error = np.zeros((len(data), 5))
    for i in range(len(data)):
        imgPath, bbox, landmarkGt,eyedist = data[i]
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
        f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]
        f_face = cv2.resize(f_face, (img_size, img_size))
        f_face = f_face.reshape((1, img_size, img_size))
        F_imgs.append(f_face)
    F_imgs = np.asarray(F_imgs)
    F_imgs = processImage(F_imgs)
    for i in range(len(data)):
        f_face = F_imgs[i]
        f_face = f_face.reshape((1, 1, img_size, img_size))
        imgPath, bbox, landmarkGt,eyedist = data[i]
        #landmarkP = getResult(img, bbox)
        F = getCNNs(level = 1)[0]
        landmarkP = F.forward(f_face)
        # real landmark
        landmarkP = bbox.reprojectLandmark(landmarkP)
        landmarkGt = bbox.reprojectLandmark(landmarkGt)
        error[i] = evaluateError(landmarkGt, landmarkP, bbox)
    return error

nameMapper = ['F_test', 'level1_test', 'level2_test', 'level3_test']

if __name__ == '__main__':
  
    t = time.clock()
    error = E()
    t = time.clock() - t

    N = len(error)
    fps = N / t
    errorMean = error.mean(0)
    # failure
    failure = np.zeros(5)
    threshold = 0.05
    for i in range(5):
        failure[i] = float(sum(error[:, i] > threshold)) / N
    # log string
    s = template % (N, t, fps, 1, errorMean[0], errorMean[1], errorMean[2], \
        errorMean[3], errorMean[4], failure[0], failure[1], failure[2], \
        failure[3], failure[4])
    print s
    logfile = 'log/1_F.log'
    with open(logfile, 'w') as fd:
        fd.write(s)
