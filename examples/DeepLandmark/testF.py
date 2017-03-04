import os, sys
import time
from functools import partial
import cv2
import numpy as np
import math
from numpy.linalg import norm
import matplotlib.pyplot as plt
from common import getCNNs
from common import getDataFromTxt,getDataFromAFLWTxt, logger
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

def evaluateError(landmarkGt, landmarkP, benchmark):
    e = np.zeros(5)
    for i in range(5):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    e = e / benchmark
    print 'landmarkGt'
    print landmarkGt
    print 'landmarkP'
    print landmarkP
    print 'error', e
    return e
     
def E(txt, isColor, test_mode, img_size, deploy_proto, caffemodel, layer_name, dataset):
    F_imgs = []
    if(isColor):
        channel = 3
    else:
        channel = 1
    if(dataset == 'LFW'):
        data = getDataFromTxt(txt)
    elif(dataset == 'AFLW'):
        data = getDataFromAFLWTxt(txt)
    error = np.zeros((len(data), 5))
    for i in range(len(data)):
        imgPath, bbox, landmarkGt,eyedist = data[i]
        if(isColor):
            img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_COLOR)
        else:
            img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        #print (img.shape[0], img.shape[1])
        f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05, img.shape[0], img.shape[1])
        #f_bbox = bbox
        #print (f_bbox.top, f_bbox.bottom, f_bbox.left, f_bbox.right)
        f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]
        f_face = cv2.resize(f_face, (img_size, img_size))
        f_face = f_face.reshape((channel, img_size, img_size))
        F_imgs.append(f_face)
    F_imgs = np.asarray(F_imgs)
    F_imgs = processImage(F_imgs, channel)
    for i in range(len(data)):
        f_face = F_imgs[i]
        f_face = f_face.reshape((1, channel, img_size, img_size))
        imgPath, bbox, landmarkGt,eyedist = data[i]
        #landmarkP = getResult(img, bbox)
        F = getCNNs(deploy_proto, caffemodel)[0]
        landmarkP = F.forward(f_face, layer_name)
        # real landmark
        landmarkP = bbox.reprojectLandmark(landmarkP)
        landmarkGt = bbox.reprojectLandmark(landmarkGt)
        eyedist = math.sqrt((landmarkGt[0][0] - landmarkGt[1][0])*(landmarkGt[0][0] - landmarkGt[1][0]) + (landmarkGt[0][1] - landmarkGt[1][1])*(landmarkGt[0][1] - landmarkGt[1][1]))
        if(test_mode):
            error[i] = evaluateError(landmarkGt, landmarkP, eyedist)
        else:
            error[i] = evaluateError(landmarkGt, landmarkP, bbox.w)
    return error

nameMapper = ['F_test', 'level1_test', 'level2_test', 'level3_test']

if __name__ == '__main__':
    assert(len(sys.argv) == 10)
    txt = sys.argv[1]
    dataset = sys.argv[2]
    isColor = int(sys.argv[3])
    test_mode = int(sys.argv[4])
    img_size = int(sys.argv[5])
    deploy_proto = sys.argv[6]
    caffemodel = sys.argv[7]
    layer_name = sys.argv[8]
    save_path = sys.argv[9]
    t = time.clock()
    error = E(txt, isColor, test_mode, img_size, deploy_proto, caffemodel, layer_name, dataset)
    t = time.clock() - t

    N = len(error)
    fps = N / t
    errorMean = error.mean(0)
    # failure
    failure = np.zeros(5)
    if(test_mode):
    	threshold = 0.1
    else:
        threshold = 0.05
    for i in range(5):
        failure[i] = float(sum(error[:, i] > threshold)) / N
    # log string
    s = template % (N, t, fps, 1, errorMean[0], errorMean[1], errorMean[2], \
        errorMean[3], errorMean[4], failure[0], failure[1], failure[2], \
        failure[3], failure[4])
    print s
    logfile = save_path
    with open(logfile, 'w') as fd:
        fd.write(s)
