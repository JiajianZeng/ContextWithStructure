import os
import math
import sys
import lmdb
import os.path 
import cv2
import numpy as np
import caffe
import argparse
from util import str2bool, log, draw_landmark_in_original_image, draw_landmark_in_cropped_face, getDataFromTxt, getDataFromTxtAFLW, BBox
from image_augmentation import flip, rotate
from lmdb_util import read_image_from_lmdb
import PIL.Image

def generate_test_lmdb(meta_txt, img_base_dir, output_dir, lmdb_name, is_color, img_size, num_landmarks=5, num_to_visualize=4, le_index=0, re_index=1):
    """
    Generate test lmdb data.
    """
    data = getDataFromTxt(meta_txt, img_base_dir, num_landmarks=num_landmarks)
    # create base dir if not existed
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output = os.path.join(output_dir, lmdb_name)

    # open image lmdb    
    in_db_img = lmdb.open(output + '_data', map_size=1e12)  
    in_txn_img = in_db_img.begin(write=True)
    
    # open landmarks lmdb
    in_db_lmk = lmdb.open(output + '_landmark', map_size=1e12)    
    in_txn_lmk = in_db_lmk.begin(write=True)

    # open bbox lmdb
    in_db_bbox = lmdb.open(output + '_bbox', map_size=1e12)
    in_txn_bbox = in_db_bbox.begin(write=True)
   
    # open eye center lmdb
    in_db_eye_center = lmdb.open(output + '_eye_center', map_size=1e12)
    in_txn_eye_center = in_db_eye_center.begin(write=True)
    
    # open eye dist lmdb
    in_db_eyedist = lmdb.open(output + '_eyedist', map_size=1e12)
    in_txn_eyedist = in_db_eyedist.begin(write=True)

    display_bbox = BBox([0, img_size, 0, img_size])
    for i, (img_path, bbox, landmark) in enumerate(data):
        # read image 
        if is_color:
           img = cv2.imread(img_path, 1)
        else:
           img = cv2.imread(img_path, 0)
        assert(img is not None)
        log("process %d, %s" % (i, img_path))

        # make sure that when the face bounding box is incorrectly labeld,
        # the coordinates, width and height computation is correct
        enlarged_bbox = bbox
        enlarged_bbox.misc_clip(img.shape[0], img.shape[1])        
        face_original = img[int(enlarged_bbox.top):int(enlarged_bbox.bottom) + 1, int(enlarged_bbox.left):int(enlarged_bbox.right) + 1]
        face_original = cv2.resize(face_original, (img_size, img_size))     
        landmark_original = enlarged_bbox.normalize_landmarks(landmark)
        # put original face image and landmark into lmdb
        in_txn_img = put_image_into_txn(in_txn_img, face_original, i)
        in_txn_lmk = put_landmark_into_txn(in_txn_lmk, landmark_original, num_landmarks, i)

        # put bbox into lmdb
        bbox_info = np.array([enlarged_bbox.right - enlarged_bbox.left, enlarged_bbox.bottom - enlarged_bbox.top])
        in_txn_bbox = put_label_into_txn(in_txn_bbox, bbox_info, 2, 1, 1, i)

        # put eye center into lmdb
        eye_center_info = np.array([landmark[le_index][0] - landmark[re_index][0], landmark[le_index][1] - landmark[re_index][1]])
        in_txn_eye_center = put_label_into_txn(in_txn_eye_center, eye_center_info, 2, 1, 1, i)

        # put eyedist into lmdb
        eyedist = math.sqrt(
	    (landmark_original[le_index][0] - landmark_original[re_index][0]) * (landmark_original[le_index][0] - landmark_original[re_index][0]) + 
	    (landmark_original[le_index][1] - landmark_original[re_index][1]) * (landmark_original[le_index][1] - landmark_original[re_index][1])
	)
        eyedist = np.asarray(eyedist)
        in_txn_eyedist = put_label_into_txn(in_txn_eyedist, eyedist, 1, 1, 1, i)

        # for debugging 
        if i < num_to_visualize:
            draw_landmark_in_cropped_face(face_original, display_bbox.denormalize_landmarks(landmark_original), os.path.join("visualization", "test_" + os.path.basename(img_path)))
            draw_landmark_in_original_image(img, enlarged_bbox, landmark, os.path.join("visualization", os.path.basename(img_path)))

        # commit the transaction per 1000 images
        if i % 1000 == 0 and i > 0:
            in_txn_img.commit()
            in_txn_lmk.commit()
            in_txn_bbox.commit()
            in_txn_eye_center.commit()
            in_txn_eyedist.commit()
            in_txn_img = in_db_img.begin(write=True)
            in_txn_lmk = in_db_lmk.begin(write=True)
            in_txn_bbox = in_db_bbox.begin(write=True)
            in_txn_eye_center = in_db_eye_center.begin(write=True)
            in_txn_eyedist = in_db_eyedist.begin(write=True)
            log("transactions committed, %d images processed." % (i))
    
    in_txn_img.commit()
    in_txn_lmk.commit()
    in_txn_bbox.commit()
    in_txn_eye_center.commit()
    in_txn_eyedist.commit()
    in_db_img.close()
    in_db_lmk.close()
    in_db_bbox.close()
    in_db_eye_center.close()
    in_db_eyedist.close()
    
def generate_training_lmdb(meta_txt, img_base_dir, output_dir, lmdb_name, is_color, img_size, num_landmarks=5, flipping=True, rotation=True, rotated_flipping=True, rotation_angles=[15,-15], num_to_visualize=4):
    """
    Generate training lmdb data.
    """
    data = getDataFromTxt(meta_txt, img_base_dir, num_landmarks=num_landmarks)
    # create base dir if not existed
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output = os.path.join(output_dir, lmdb_name)

    # open image lmdb    
    in_db_img = lmdb.open(output + '_data', map_size=1e12)  
    in_txn_img = in_db_img.begin(write=True)
    
    # open landmarks lmdb
    in_db_lmk = lmdb.open(output + '_landmark', map_size=1e12)    
    in_txn_lmk = in_db_lmk.begin(write=True)
    
    count = 0
    shuffle_idx = np.random.permutation(500000)
    display_bbox = BBox([0, img_size, 0, img_size])
    for i, (img_path, bbox, landmark) in enumerate(data):
        # read image 
        if is_color:
           img = cv2.imread(img_path, 1)
        else:
           img = cv2.imread(img_path, 0)
        assert(img is not None)
        log("process %d, %s" % (i, img_path))

        # make sure that when the face bounding box is incorrectly labeld,
        # the coordinates, width and height computation is correct
        enlarged_bbox = bbox
        enlarged_bbox.misc_clip(img.shape[0], img.shape[1])
        
        face_original = img[int(enlarged_bbox.top):int(enlarged_bbox.bottom) + 1, int(enlarged_bbox.left):int(enlarged_bbox.right) + 1]
        face_original = cv2.resize(face_original, (img_size, img_size))     
        landmark_original = enlarged_bbox.normalize_landmarks(landmark)
        # put original face image and landmark into lmdb
        in_txn_img = put_image_into_txn(in_txn_img, face_original, shuffle_idx[count])
        in_txn_lmk = put_landmark_into_txn(in_txn_lmk, landmark_original, num_landmarks, shuffle_idx[count])
        count += 1
        # for debugging 
        if i < num_to_visualize:
            draw_landmark_in_cropped_face(face_original, display_bbox.denormalize_landmarks(landmark_original), os.path.join("visualization", "original_" + os.path.basename(img_path)))
            draw_landmark_in_original_image(img, enlarged_bbox, landmark, os.path.join("visualization", os.path.basename(img_path)))
        
        # flipping
        if flipping is True:
            # horizontal flipping 
            face_flipped, bbox_flipped, landmark_flipped = flip(img, enlarged_bbox, landmark)
            face_flipped = cv2.resize(face_flipped, (img_size, img_size))
            landmark_flipped = bbox_flipped.normalize_landmarks(landmark_flipped)
            # put flipped face image and landmark into lmdb
            in_txn_img = put_image_into_txn(in_txn_img, face_flipped, shuffle_idx[count])
            in_txn_lmk = put_landmark_into_txn(in_txn_lmk, landmark_flipped, num_landmarks, shuffle_idx[count])
            count += 1
            # for debugging
            if i < num_to_visualize:
                draw_landmark_in_cropped_face(face_flipped, display_bbox.denormalize_landmarks(landmark_flipped), os.path.join("visualization", "flipped_" + os.path.basename(img_path)))
        
        # rotation
        if rotation is True:            
	    # rotate with probability 100%
	    for alpha in rotation_angles:
		if np.random.rand() > -1:
		    img_rotated, face_rotated, landmark_rotated = rotate(img, enlarged_bbox, \
		        landmark, alpha)
		    landmark_rotated = enlarged_bbox.normalize_landmarks(landmark_rotated)
		    face_rotated = cv2.resize(face_rotated, (img_size, img_size))
                    # put rotated face image and landmark into lmdb
                    in_txn_img = put_image_into_txn(in_txn_img, face_rotated, shuffle_idx[count])
                    in_txn_lmk = put_landmark_into_txn(in_txn_lmk, landmark_rotated, num_landmarks, shuffle_idx[count])
                    count += 1
		    # for debugging
		    if i < num_to_visualize:
		        draw_landmark_in_cropped_face(face_rotated, display_bbox.denormalize_landmarks(landmark_rotated), os.path.join("visualization", "rotated_" + str(alpha) + "_" + os.path.basename(img_path)))
		
		    # horizontal flipping after rotation
                    if rotated_flipping is True:
		        face_flipped, bbox_flipped, landmark_flipped = flip(img_rotated, enlarged_bbox, enlarged_bbox.denormalize_landmarks(landmark_rotated))
		        face_flipped = cv2.resize(face_flipped, (img_size, img_size))
		        landmark_flipped = bbox_flipped.normalize_landmarks(landmark_flipped)
                        # put rotated flipping face image and landmark into lmdb
                        in_txn_img = put_image_into_txn(in_txn_img, face_flipped, shuffle_idx[count])
                        in_txn_lmk = put_landmark_into_txn(in_txn_lmk, landmark_flipped, num_landmarks, shuffle_idx[count])
                        count += 1
		        # for debugging
		        if i < num_to_visualize:
		            draw_landmark_in_cropped_face(face_flipped, display_bbox.denormalize_landmarks(landmark_flipped), os.path.join("visualization", "rotated_flip_" + str(alpha) + "_" + os.path.basename(img_path)))

        # commit the transaction per 1000 images
        if i % 1000 == 0 and i > 0:
            in_txn_img.commit()
            in_txn_lmk.commit()
            in_txn_img = in_db_img.begin(write=True)
            in_txn_lmk = in_db_lmk.begin(write=True)
            log("transactions committed, %d images processed." % (i))
    in_txn_img.commit()
    in_txn_lmk.commit()
    in_db_img.close()   
    in_db_lmk.close()   
    log("number of total generated entries: %d" % (count))

def put_image_into_txn(in_txn, in_img, i):
    """
    Put image into txn, using RGB order.
    """
    # h * w * c -> c * h * w
    out_img = np.transpose(in_img, (2, 0, 1))
    # BGR -> RGB
    out_img = out_img[::-1,:,:]  
    im_dat = caffe.io.array_to_datum(out_img)  
    in_txn.put('{:0>10d}'.format(i), im_dat.SerializeToString())
    return in_txn

def put_landmark_into_txn(in_txn, in_lmk, num_landmarks, i):
    """
    Put landmark into txn.
    """
    out_lmk = in_lmk.reshape(2 * num_landmarks)
    datum = caffe.proto.caffe_pb2.Datum()  
    datum.channels = out_lmk.shape[0]
    datum.height = 1
    datum.width = 1
    datum.float_data.extend(out_lmk.astype(float).flat)
    in_txn.put("{:0>10d}".format(i), datum.SerializeToString())
    return in_txn

def put_label_into_txn(in_txn, label, channels, height, width, i):
    """
    Put label into txn.
    """
    datum = caffe.proto.caffe_pb2.Datum()  
    datum.channels = channels
    datum.height = height
    datum.width = width
    datum.float_data.extend(label.astype(float).flat)
    in_txn.put("{:0>10d}".format(i), datum.SerializeToString())
    return in_txn
    
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
                        default=True, type=str2bool)
    # image size 
    parser.add_argument('--img_size',
                        help='image size',
                        default=224, type=int)
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
    # perform flipping or not
    parser.add_argument('--flipping',
                        help='perform flipping or not',
                        default=True, type=str2bool)
    # perform rotation or not
    parser.add_argument('--rotation',
                        help='perform rotation or not',
                        default=True, type=str2bool)
    # perform rotated flipping or not
    parser.add_argument('--rotated_flipping',
                        help='perform rotated flipping or not',
                        default=True, type=str2bool)
    # number of images to visualize to test whether the generation process is correct or not
    parser.add_argument('--num_to_visualize',
                        help='number of images to visualize to test whether the generation process is correct or not',
                        default=4, type=int)
    # rotation angles for data augmentation
    parser.add_argument('--rotation_angles', action='append',
                        help='rotation angles for data augmention (in degrees)',
                        default=None, type=int)
    # whether resize an existing lmdb to another size
    parser.add_argument('--resize_lmdb',
                        help='whether resize an existing lmdb to another size',
                        default=False, type=str2bool)
    # whether whitening an existing lmdb
    parser.add_argument('--whitening_lmdb', 
                        help='whether whitening an existing lmdb', 
                        default=False, type=str2bool)
    # lmdb to compute mean and std
    parser.add_argument('--lmdb2compute_mean_and_std',
                        help='lmdb to compute mean and std',
                        default=None, type=str)
    # the original lmdb data file
    parser.add_argument('--original_lmdb',
                        help='the original lmdb data file (resize_lmdb flag must be set to true)',
                        default=None, type=str)
    # generate test stage data or not
    parser.add_argument('--test_stage',
                        help='whether the generated lmdb data is for test stage or not',
                        default=False, type=str2bool)
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
    num_landmarks = args.num_landmarks
    le_index = args.le_index
    re_index = args.re_index
    flipping = args.flipping
    rotation = args.rotation
    rotated_flipping = args.rotated_flipping
    num_to_visualize = args.num_to_visualize
    if args.rotation_angles is not None:
        rotation_angles = args.rotation_angles
    else:
        rotation_angles = [15, -15]
    
    resize_lmdb = args.resize_lmdb
    whitening_lmdb = args.whitening_lmdb
    lmdb2compute_mean_and_std = args.lmdb2compute_mean_and_std
    original_lmdb = args.original_lmdb
    test_stage = args.test_stage
    
    # resize an existing lmdb to another size
    if resize_lmdb and original_lmdb is not None:
        resize_existing_lmdb(original_lmdb, img_size, output_dir, lmdb_prefix)
    elif whitening_lmdb and original_lmdb is not None:
        subtract_mean_divide_std(lmdb2compute_mean_and_std, original_lmdb, output_dir, lmdb_prefix, is_color)
    # generate lmdb
    elif test_stage is not True:
        generate_training_lmdb(meta_file, img_base_dir, output_dir, lmdb_prefix, is_color, img_size, num_landmarks=num_landmarks, flipping=flipping, rotation=rotation, rotated_flipping=rotated_flipping, rotation_angles=rotation_angles, num_to_visualize=num_to_visualize)
    else:
        generate_test_lmdb(meta_file, img_base_dir, output_dir, lmdb_prefix, is_color, img_size, num_landmarks=num_landmarks, num_to_visualize=num_to_visualize, le_index=le_index, re_index=re_index)
        
