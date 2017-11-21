import os
import math
import sys
import lmdb
import os.path 
import cv2
import numpy as np
import caffe
import argparse
from util import log, draw_landmark_in_original_image, draw_landmark_in_cropped_face, getDataFromTxt, getDataFromTxtAFLW, BBox
from image_augmentation import flip, rotate
from lmdb_util import read_image_from_lmdb
import PIL.Image

def generate_lmdb(meta_txt, img_base_dir, output_dir, lmdb_name, is_color, img_size, augment=False, num_landmarks=5, le_index=0, re_index=1, rotation_angles=[5,-5], num_to_visualize=4):
    """
    Generate lmdb data.
    """
    # for AFLW dataset
    data = getDataFromTxtAFLW(meta_txt, img_base_dir, num_landmarks=num_landmarks)
    
    # for other datasets that does not need visibility information
    # data = getDataFromTxt(meta_txt, img_base_dir, num_landmarks=num_landmarks)
    imgs = []
    landmarks = []
    eyedists = []
    # create base dir if not existed
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output = os.path.join(output_dir, lmdb_name)

    # write image to lmdb    
    in_db_img = lmdb.open(output + '_data', map_size=1e12)  
    in_txn_img = in_db_img.begin(write=True)
    
    
    # write landmarks to lmdb
    in_db_lmk = lmdb.open(output + '_landmark', map_size=1e12)    
    in_txn_lmk = in_db_lmk.begin(write=True)
    

    # write eyedist to lmdb
    in_db_dist = lmdb.open(output + '_eyedist', map_size=1e12)  
    in_txn_dist = in_db_dist.begin(write=True)

    # write visibility to lmdb
    in_db_v = lmdb.open(output + '_visibility', map_size=1e12)
    in_txn_v = in_db_v.begin(write=True)

    # write eye center offset to lmdb
    in_db_eye_center = lmdb.open(output + '_eye_center', map_size=1e12)
    in_txn_eye_center = in_db_eye_center.begin(write=True)
     

    for i, (img_path, bbox, landmark, v) in enumerate(data):
        # read image 
        if is_color:
           channel = 3
           img = cv2.imread(img_path, 1)
        else:
           channel = 1
           img = cv2.imread(img_path, 0)
        assert(img is not None)
        log("process %d,  %s" % (i ,img_path))

        # enlarge face bounding box and crop, resize the face  
        # enlarged_bbox = bbox.enlarge(0.05, 0.05, 0.05, 0.05)
        # if not enlarged_bbox.valid(img.shape[0], img.shape[1]):
        enlarged_bbox = bbox
        # make sure that when the face bounding box is incorrectly labeld,
        # the coordinates, width and height computation is correct
        enlarged_bbox.misc_clip(img.shape[0], img.shape[1])
        
        face_original = img[int(enlarged_bbox.top):int(enlarged_bbox.bottom) + 1, int(enlarged_bbox.left):int(enlarged_bbox.right) + 1]
        face_original = cv2.resize(face_original, (img_size, img_size))
             
        # normalize landmark to range [0, 1]
        landmark_original = enlarged_bbox.normalize_landmarks(landmark)
        # put into container
        # h * w * c -> c * h * w
        im = np.transpose(face_original, (2, 0, 1))
        # BGR -> RGB
        im = im[::-1,:,:]  
        im_dat = caffe.io.array_to_datum(im)  
        in_txn_img.put('{:0>10d}'.format(i), im_dat.SerializeToString())  
        
        # write landmark 
        lmk = landmark_original.reshape(2 * num_landmarks)
        datum = caffe.proto.caffe_pb2.Datum()  
        datum.channels = lmk.shape[0]
        datum.height = 1
        datum.width = 1
        datum.float_data.extend(lmk.astype(float).flat)
        in_txn_lmk.put("{:0>10d}".format(i), datum.SerializeToString()) 
        
        # write visibility, only for AFLW dataset
        extend_v = np.zeros(2 * num_landmarks, dtype=np.int)
        for i in range(num_landmarks):
            extend_v[2 * i] = v[i]
            extend_v[2 * i + 1] = v[i]
        datum = caffe.proto.caffe_pb2.Datum()  
        datum.channels = extend_v.shape[0]
        datum.height = 1
        datum.width = 1
        datum.float_data.extend(extend_v.astype(float).flat)
        in_txn_v.put("{:0>10d}".format(i), datum.SerializeToString()) 
        
        eyedist = math.sqrt(
            (landmark_original[le_index][0] - landmark_original[re_index][0]) * (landmark_original[le_index][0] - landmark_original[re_index][0]) + 
            (landmark_original[le_index][1] - landmark_original[re_index][1]) * (landmark_original[le_index][1] - landmark_original[re_index][1])
            )
        eyedist = np.asarray(eyedist)
        # eyedist = np.array([enlarged_bbox.right - enlarged_bbox.left, enlarged_bbox.bottom - enlarged_bbox.top])
        datum = caffe.proto.caffe_pb2.Datum()  
        datum.channels = 1
        datum.height = 1
        datum.width = 1
        datum.float_data.extend(eyedist.astype(float).flat)
        in_txn_dist.put("{:0>10d}".format(i), datum.SerializeToString())  


        eye_center_offset = np.array([landmark[le_index][0] - landmark[re_index][0], landmark[le_index][1] - landmark[re_index][1]])
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 2
        datum.height = 1
        datum.width = 1
        datum.float_data.extend(eye_center_offset.astype(float).flat)
        in_txn_eye_center.put("{:0>10d}".format(i), datum.SerializeToString())
        
        ## for debug 
        if i < num_to_visualize:
            bbox = BBox([0, img_size, 0, img_size])
            draw_landmark_in_cropped_face(face_original, bbox.denormalize_landmarks(landmark_original), os.path.join("visualization", "original_" + os.path.basename(img_path)))
            draw_landmark_in_original_image(img, enlarged_bbox, landmark, os.path.join("visualization", os.path.basename(img_path)))
        
        # data augmentation
        if augment:
            # horizontally flip 
            '''face_flipped, bbox_flipped, landmark_flipped = flip(img, enlarged_bbox, landmark)
            face_flipped = cv2.resize(face_flipped, (img_size, img_size))
            landmark_flipped = bbox_flipped.normalize_landmarks(landmark_flipped)
            
            imgs.append(np.transpose(face_flipped, (2, 0, 1)))
            landmarks.append(landmark_flipped.reshape(2 * num_landmarks))
            eyedists.append(eyedist)
            ## for debug
            if i < num_to_visualize:
                bbox = BBox([0, img_size, 0, img_size])
                draw_landmark_in_cropped_face(face_flipped, bbox.denormalize_landmarks(landmark_flipped), os.path.join("visualization", "flipped_" + os.path.basename(img_path)))
            '''
            # rotate with probability 100%
            for alpha in rotation_angles:
                if np.random.rand() > -1:
                    img_rotated, face_rotated, landmark_rotated = rotate(img, enlarged_bbox, \
                        landmark, alpha)
                    # normalize the landmark to range [0,1]
                    landmark_rotated = enlarged_bbox.normalize_landmarks(landmark_rotated)
                    # resize face 
                    face_rotated = cv2.resize(face_rotated, (img_size, img_size))
                    imgs.append(np.transpose(face_rotated, (2, 0, 1)))
                    landmarks.append(landmark_rotated.reshape(2 * num_landmarks))
                    eyedists.append(eyedist)
                    ## for debug
                    if i < num_to_visualize:
                        bbox = BBox([0, img_size, 0, img_size])
                        draw_landmark_in_cropped_face(face_rotated, bbox.denormalize_landmarks(landmark_rotated), os.path.join("visualization", "rotated_" + str(alpha) + "_" + os.path.basename(img_path)))
                
                    # horizontally flip the rotated face
                    '''
                    face_flipped, bbox_flipped, landmark_flipped = flip(img_rotated, enlarged_bbox, enlarged_bbox.denormalize_landmarks(landmark_rotated))
                    face_flipped = cv2.resize(face_flipped, (img_size, img_size))
                    landmark_flipped = bbox_flipped.normalize_landmarks(landmark_flipped)
                    imgs.append(np.transpose(face_flipped, (2, 0, 1)))
                    landmarks.append(landmark_flipped.reshape(2 * num_landmarks))
                    eyedists.append(eyedist)

                    ## for debug
                    if i < num_to_visualize:
                        bbox = BBox([0, img_size, 0, img_size])
                        draw_landmark_in_cropped_face(face_flipped, bbox.denormalize_landmarks(landmark_flipped), os.path.join("visualization", "flipped_rotated_" + str(alpha) + "_" + os.path.basename(img_path)))
                    '''
    in_txn_img.commit()
    in_txn_lmk.commit()
    in_txn_dist.commit()
    in_txn_v.commit()
    in_txn_eye_center.commit()

    in_db_img.close()   
    in_db_lmk.close()   
    in_db_dist.close()   
    in_db_v.close()     
    in_db_eye_center.close()      
    log('number of total generated images: %s' % str(len(imgs)))
        
    
 

    

def resize_existing_lmdb(original_lmdb_data, img_size, output_dir, lmdb_prefix):
    """
    Resize an existing lmdb to another size (spatial). For example (N, C, 224, 224) -> (N, C, 32, 32). Using this function we can generate 
    another spatial sized image lmdb easily.
    
    param:
    -original_lmdb_data, path to the original lmdb data
    -img_size, new spatial size of the image 
    -output_dir, where to save the generated lmdb file
    -lmdb_prefix, prefix for the generated lmdb file
    """
    # each element in list imgs is of shape (c, h, w) and is in RGB order
    imgs = read_image_from_lmdb(original_lmdb_data, np.iinfo(np.int32).max, '.', vis=False, print_info=True)
    # create base dir if not existed
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output = os.path.join(output_dir, lmdb_prefix)
    log("generate %s" % output)

    lmdb_env = lmdb.open(output + '_data', map_size=1e12)
    with lmdb_env.begin(write=True) as lmdb_txn:
        for idx, im in enumerate(imgs):  
            # c * h * w -> h * w * c
            im = np.transpose(im, (1, 2, 0))
            # resize
            im = cv2.resize(im, (img_size, img_size))
            # h * w * c -> c * h * w
            im = np.transpose(im, (2, 0, 1))  
            im_dat = caffe.io.array_to_datum(im)  
            lmdb_txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())  
    lmdb_env.close()

def subtract_mean_divide_std(lmdb2compute_mean_and_std, original_lmdb_data, output_dir, lmdb_prefix, is_color=True):
    """
    For a set of images, subtract the mean image from each image and divide it by the std image.

    param:
    -original_lmdb_data, path to the original lmdb data
    -output_dir, where to save the generated lmdb data
    -lmdb_prefix, prefix for the generated lmdb file
    """
    # each element in list imgs is of shape (c, h, w) and is in RGB order
    imgs = read_image_from_lmdb(lmdb2compute_mean_and_std, np.iinfo(np.int32).max, '.', vis=False, print_info=True)
    imgs = np.asarray(imgs)
    if is_color:
        print "computing mean image..."
        m_r = np.mean(imgs[:, 0, :, :], axis=0)
        m_g = np.mean(imgs[:, 1, :, :], axis=0)
        m_b = np.mean(imgs[:, 2, :, :], axis=0)
        print "finish computing mean image."
        print "computing std image..."
        std_r = np.std(imgs[:, 0, :, :], axis=0)
        std_g = np.std(imgs[:, 1, :, :], axis=0)
        std_b = np.std(imgs[:, 2, :, :], axis=0)
        print "finish computing std image."

    imgs2whitening = read_image_from_lmdb(original_lmdb_data, np.iinfo(np.int32).max, '.', vis=False, print_info=True)
    imgs2whitening = np.asarray(imgs2whitening)
    for i, _ in enumerate(imgs2whitening):
        imgs2whitening[i][0] = (imgs2whitening[i][0] - m_r) / std_r
        imgs2whitening[i][1] = (imgs2whitening[i][1] - m_g) / std_g
        imgs2whitening[i][2] = (imgs2whitening[i][2] - m_b) / std_b
        print "whitening %d/%d" % (i, len(imgs2whitening))
    # create base dir if not existed
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output = os.path.join(output_dir, lmdb_prefix)
    log("generate %s" % output)
    
    lmdb_env = lmdb.open(output + '_data', map_size=1e12)
    with lmdb_env.begin(write=True) as lmdb_txn:
        for idx, im in enumerate(imgs2whitening):  
            im_dat = caffe.io.array_to_datum(im)  
            lmdb_txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())  
    lmdb_env.close()
    
def parse_args():
    parser = argparse.ArgumentParser(description='Generate lmdb file of the image as well as its corresponding landmark and eyedist.')
    # meta file of the dataset
    parser.add_argument('--meta_file',
                        help='meta file of the dataset',
                        default=None, type=str)
    # base directory of the image 
    parser.add_argument('--img_base_dir',
                        help='base directory of the image',
                        default=None, type=str)
    # output directory of the lmdb files
    parser.add_argument('--output_dir',
                        help='output directory of the lmdb files',
                        default=None, type=str)
    # lmdb file prefix
    parser.add_argument('--lmdb_prefix',
                        help='lmdb file prefix',
                        default=None, type=str)
    # generate image lmdb in color space or not
    parser.add_argument('--is_color',
                        help='generate image lmdb in color space or not',
                        default=True, type=bool)
    # image size 
    parser.add_argument('--img_size',
                        help='image size',
                        default=224, type=int)
    # augment or not
    parser.add_argument('--augment',
                        help='augment or not',
                        default=False, type=bool)
    # number of landmarks
    parser.add_argument('--num_landmarks',
                        help='number of landmarks',
                        default=5, type=int)
    # left eye index
    parser.add_argument('--le_index',
                         help='left eye index',
                         default=0, type=int)
    # right eye index
    parser.add_argument('--re_index',
                         help='right eye index',
                         default=1, type=int)
    # number of images to visualize to test whether the generation process is correct or not
    parser.add_argument('--num_to_visualize',
                         help='number of images to visualize to test whether the generation process is correct or not',
                         default=4, type=int)
    # rotation angles for data augmentation
    parser.add_argument('--rotation_angles', action='append',
                        help='rotation angles for data augmention (in degrees)',
                        default=[5,-5], type=int)
    # whether resize an existing lmdb to another size
    parser.add_argument('--resize_lmdb',
                         help='whether resize an existing lmdb to another size',
                         default=False, type=bool)
    # whether whitening an existing lmdb
    parser.add_argument('--whitening_lmdb', 
                         help='whether whitening an existing lmdb', 
                         default=False, type=bool)
    # lmdb to compute mean and std
    parser.add_argument('--lmdb2compute_mean_and_std',
                         help='lmdb to compute mean and std',
                         default=None, type=str)
    # the original lmdb data file
    parser.add_argument('--original_lmdb',
                         help='the original lmdb data file (resize_lmdb flag must be set to true)',
                         default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':
    # parse args
    args = parse_args()
    meta_file = args.meta_file
    img_base_dir = args.img_base_dir
    output_dir = args.output_dir
    lmdb_prefix = args.lmdb_prefix
    is_color = args.is_color
    img_size = args.img_size
    augment = args.augment
    num_landmarks = args.num_landmarks
    le_index = args.le_index
    re_index = args.re_index
    num_to_visualize = args.num_to_visualize
    rotation_angles = args.rotation_angles
    
    resize_lmdb = args.resize_lmdb
    whitening_lmdb = args.whitening_lmdb
    lmdb2compute_mean_and_std = args.lmdb2compute_mean_and_std
    original_lmdb = args.original_lmdb
    
    # resize an existing lmdb to another size
    if resize_lmdb and original_lmdb is not None:
        resize_existing_lmdb(original_lmdb, img_size, output_dir, lmdb_prefix)
    elif whitening_lmdb and original_lmdb is not None:
        subtract_mean_divide_std(lmdb2compute_mean_and_std, original_lmdb, output_dir, lmdb_prefix, is_color)
    # generate lmdb
    else:
        generate_lmdb(meta_file, img_base_dir, output_dir, lmdb_prefix, is_color, img_size, augment=augment, num_landmarks=num_landmarks, le_index=le_index, re_index=re_index, rotation_angles=rotation_angles, num_to_visualize=num_to_visualize)
    
    ## train_txt = os.path.join('/share/disk/zengjiajian_dataset/LFW_NET', 'trainImageList.txt')
    ## generate_lmdb(train_txt, '/share/disk/zengjiajian_dataset/LFW_NET', 'dataset/train', 'lfw_net_224x224_rgb', True, 224, augment=True)

    ## test_txt = os.path.join('/share/disk/zengjiajian_dataset/LFW_NET', 'testImageList.txt')
    ## generate_lmdb(test_txt, '/share/disk/zengjiajian_dataset/LFW_NET', 'dataset/test', 'lfw_net_224x224_rgb', True, 224, augment=False)

    ## test_txt = os.path.join('/share/disk/zengjiajian_dataset/MTFL_TEST', 'correct_test.txt')
    ## generate_lmdb(test_txt, '/share/disk/zengjiajian_dataset/MTFL_TEST', 'dataset/test', 'mtfl_test_224x224_rgb', True, 224, augment=False, num_to_visualize=15)

    ## train_txt = os.path.join('/share/disk/zengjiajian_dataset/AFLW_FULL', 'trainImageList.txt')
    ## generate_lmdb(train_txt, '/share/disk/zengjiajian_dataset/AFLW_FULL', 'dataset/train', 'aflw_full_224x224_rgb', True, 224, augment=True, num_landmarks=19, num_to_visualize=15)
    # Done
