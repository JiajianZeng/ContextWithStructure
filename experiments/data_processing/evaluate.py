import os, sys
import numpy as np
import time
from numpy.linalg import norm
import argparse
import caffe

from cnn_util import CNN
from lmdb_util import read_image_from_lmdb, read_label_from_lmdb
from util import BBox, draw_landmark_in_cropped_face


template_5_points = \
'''################## Summary ################## 
Network: %s 
Dataset: %s 
Number of test images: %d 
FPS: %.03f 
Normalizer: %s 
Failure threshold: %.03f 
Mean Error: %f
    Left Eye       = %f 
    Right Eye      = %f 
    Nose           = %f 
    Left Mouth     = %f 
    Right Mouth    = %f 
Failure: %f
    Left Eye       = %f 
    Right Eye      = %f 
    Nose           = %f 
    Left Mouth     = %f 
    Right Mouth    = %f 
'''

template_19_points = \
'''################## Summary ################## 
Network: %s 
Dataset: %s 
Number of test images: %d 
FPS: %.03f 
Normalizer: %s 
Failure threshold: %.03f 
Mean Error: %f
    k1 = %f 
    k2 = %f 
    k3 = %f 
    k4 = %f 
    k5 = %f 
    k6 = %f 
    k7 = %f 
    k8 = %f 
    k9 = %f 
    k10 = %f 
    k11 = %f 
    k12 = %f
    k13 = %f  
    k14 = %f 
    k15 = %f 
    k16 = %f 
    k17 = %f 
    k18 = %f 
    k19 = %f 
Failure: %f
    k1 = %f 
    k2 = %f 
    k3 = %f 
    k4 = %f 
    k5 = %f 
    k6 = %f 
    k7 = %f 
    k8 = %f 
    k9 = %f 
    k10 = %f 
    k11 = %f 
    k12 = %f
    k13 = %f  
    k14 = %f 
    k15 = %f 
    k16 = %f 
    k17 = %f 
    k18 = %f 
    k19 = %f 
'''


def compute_normed_error(landmark_gt, landmark_pre, normalizer, num_landmarks=5, print_info=True):
    """
    Compute normalized error for a test sample.
    param:
    -landmark_gt, of shape (N, 2)
    -landmark_pre, of shape (N, 2)
    """
    normed_error = np.zeros(num_landmarks)
    for i in range(num_landmarks):
        # L2-norm
        normed_error[i] = norm(landmark_gt[i] - landmark_pre[i]) / normalizer
    if print_info:
        print '##### ground-truth, predicted landmark and normalized error #####'
        print landmark_gt
        print landmark_pre
        print normed_error
    return normed_error

def compute_normed_error_by_width(landmark_gt, landmark_pre, img_size, num_landmarks=5, print_info=True):
    """
    Compute normalized error for a test sample normalized by bounding box width.
    param:
    -landmark_gt, of shape (N, 2)
    -landmark_pre, of shape (N, 2)
    -img_size, of shape (2)
    """
    normed_error = np.zeros(num_landmarks)
    for i in range(num_landmarks):
        # L2-norm
        gt = landmark_gt[i] * img_size
        pre = landmark_pre[i] * img_size
        normed_error[i] = norm(gt - pre) / img_size[0]
    if print_info:
        print '##### ground-truth, predicted landmark, image size and normalized error #####'
        print landmark_gt
        print landmark_pre
        print img_size
        print normed_error
    return normed_error
     
def evaluate(lmdb_data, lmdb_landmark, lmdb_eyedist, mean_file, num_landmarks, network, caffemodel, caffe_mode='GPU', gpu_id=0, threshold=0.1, input_layer="data", output_layer="fc8", print_info=True, use_width=False):
    """
    Evalute a specified test dataset on a specific network.
    return:
    -normalized mean error
    =failure rate
    -fps
    -number of images
    """
    # each element in list images has shape (c, h, w) already
    images = read_image_from_lmdb(lmdb_data, np.iinfo(np.int32).max, '.', vis=False, print_info=print_info)
    # each element in list landmarks_gt should be reshaped to (num_landmarks, 2) before usage
    landmarks_gt = read_label_from_lmdb(lmdb_landmark, np.iinfo(np.int32).max, print_info)
    # each element in list eyedists should be reshaped to (1) before usage
    eyedists = read_label_from_lmdb(lmdb_eyedist, np.iinfo(np.int32).max, print_info)

    # load mean image file (.binaryproto)
    blob = caffe.proto.caffe_pb2.BlobProto()
    with open(mean_file, 'rb') as fd:
        mean_file_data = fd.read()
    blob.ParseFromString(mean_file_data)
    mean_image = np.array(caffe.io.blobproto_to_array(blob))
    print mean_image
    
    cnn = CNN(network, caffemodel, caffe_mode, gpu_id)
    normed_error = np.zeros((len(images), num_landmarks))
    # forward each image
    t = time.clock()
    for i in range(len(images)):
        # expand (c, h, w)-> (1, c, h, w)
        # landmark_pre has shape (num_landmarks, 2)
        # subtract mean image
        landmark_pre = cnn.forward(np.subtract(np.expand_dims(images[i], 0), mean_image), input_layer=input_layer, output_layer=output_layer)
        '''
        # for debug, visualizing the 
        if i == 0:
            # c * h * w -> h * w * c
            img = np.transpose(images[i], (1,2,0))
            # RGB->BGR
            img = img[:, :, ::-1]
            bbox = BBox([0, img.shape[1], 0, img.shape[0]]) 
            draw_landmark_in_cropped_face(img, bbox.denormalize_landmarks(landmark_pre), '0.jpg')
        '''
        if use_width:
            normed_error[i] = compute_normed_error_by_width(landmarks_gt[i].reshape(num_landmarks, 2), landmark_pre, eyedists[i].reshape(2), num_landmarks, print_info)
        else:
            normed_error[i] = compute_normed_error(landmarks_gt[i].reshape(num_landmarks, 2), landmark_pre, eyedists[i].reshape(2), num_landmarks, print_info)
    t = time.clock() - t
    
    # failure rate
    failure_rate = np.zeros(num_landmarks)
    for i in range(num_landmarks):
        failure_rate[i] = float(np.sum(normed_error[:, i] > threshold)) / len(images)
    # normalized mean error
    normed_mean_error = normed_error.mean(0)
    return normed_mean_error, failure_rate, len(images) / t, len(images)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a specified dataset on a specific network.')
    # image lmdb
    parser.add_argument('--lmdb_data',
                        help='image lmdb file',
                        default=None, type=str)
    # landmark lmdb
    parser.add_argument('--lmdb_landmark',
                        help='landmark lmdb file',
                        default=None, type=str)
    # eyedist lmdb
    parser.add_argument('--lmdb_eyedist',
                        help='eyedist lmdb file',
                        default=None, type=str)
    # mean image file
    parser.add_argument('--mean_file',
                        help='mean image file, in general case, you should use mean image file which used by your training',
                        default=None, type=str)
    # number of landmarks
    parser.add_argument('--num_landmarks',
                        help='number of landmarks',
                        choices=[5, 19],
                        default=5, type=int)
    # network
    parser.add_argument('--network',
                        help='network',
                        default=None, type=str)
    # caffemodel
    parser.add_argument('--caffemodel',
                        help='caffemodel',
                        default=None, type=str)
    # caffe mode, GPU or CPU
    parser.add_argument('--caffe_mode',
                        help='caffe mode [GPU or CPU]',
                        choices=['GPU', 'CPU'],
                        default='GPU', type=str)
    # gpu id 
    parser.add_argument('--gpu_id',
                        help='gpu id',
                        default=0, type=int)
    # failure rate threshold
    parser.add_argument('--threshold',
                         help='failure rate threshold',
                         default=0.1, type=float)
    # input layer
    parser.add_argument('--input_layer',
                        help='input layer of the network',
                        default='data', type=str)
    # output layer
    parser.add_argument('--output_layer',
                        help='output layer of the network',
                        default='fc8', type=str)
    # print stat info or not
    parser.add_argument('--print_info',
                        help='print stat info or not',
                        default=True, type=bool)
    # normalized by bounding box width or not
    parser.add_argument('--use_width',
                        help='normalized by bounding box width or not',
                        default=False, type=bool)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()
    


if __name__ == '__main__':
    # parse args
    args = parse_args()
    lmdb_data = args.lmdb_data
    lmdb_landmark = args.lmdb_landmark
    lmdb_eyedist = args.lmdb_eyedist
    mean_file = args.mean_file
    num_landmarks = args.num_landmarks
    network = args.network
    caffemodel = args.caffemodel
    caffe_mode = args.caffe_mode
    gpu_id = args.gpu_id
    threshold = args.threshold
    input_layer = args.input_layer
    output_layer = args.output_layer
    print_info = args.print_info
    use_width = args.use_width
     
    normed_mean_error, failure_rate, fps, num_images = evaluate(lmdb_data, lmdb_landmark, lmdb_eyedist, mean_file, num_landmarks, network, caffemodel, caffe_mode, gpu_id, threshold, input_layer, output_layer, print_info, use_width=use_width)    
    # format evaluation info
    if num_landmarks == 5:
        eval_info = template_5_points % (os.path.basename(network), 
                                         os.path.basename(lmdb_data),
                                         num_images, 
                                         fps, 
                                         'bi-ocular distance',
                                         threshold, 
                                         normed_mean_error.mean(0),
                                         normed_mean_error[0], normed_mean_error[1], normed_mean_error[2], normed_mean_error[3], normed_mean_error[4],
                                         failure_rate.mean(0),
                                         failure_rate[0], failure_rate[1], failure_rate[2], failure_rate[3], failure_rate[4])
    elif num_landmarks == 19:
        eval_info = template_19_points % (os.path.basename(network), 
                                         os.path.basename(lmdb_data),
                                         num_images, 
                                         fps, 
                                         'bi-ocular distance',
                                         threshold, 
                                         normed_mean_error.mean(0),
                                         normed_mean_error[0], normed_mean_error[1], normed_mean_error[2], normed_mean_error[3], normed_mean_error[4],
                                         normed_mean_error[5], normed_mean_error[6], normed_mean_error[7], normed_mean_error[8], normed_mean_error[9],
                                         normed_mean_error[10], normed_mean_error[11], normed_mean_error[12], normed_mean_error[13], normed_mean_error[14],
                                         normed_mean_error[15], normed_mean_error[16], normed_mean_error[17], normed_mean_error[18],
                                         failure_rate.mean(0),
                                         failure_rate[0], failure_rate[1], failure_rate[2], failure_rate[3], failure_rate[4],
                                         failure_rate[5], failure_rate[6], failure_rate[7], failure_rate[8], failure_rate[9],
                                         failure_rate[10], failure_rate[11], failure_rate[12], failure_rate[13], failure_rate[14],
                                         failure_rate[15], failure_rate[16], failure_rate[17], failure_rate[18])
    print eval_info
