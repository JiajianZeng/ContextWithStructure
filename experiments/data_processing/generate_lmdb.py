import os
import math
import sys
import lmdb
import os.path 
import cv2
import numpy as np
import caffe
from util import log, draw_landmark_in_original_image, draw_landmark_in_cropped_face, getDataFromTxt,BBox
from image_augmentation import flip, rotate

def generate_lmdb(meta_txt, img_base_dir, output_dir, lmdb_name, is_color, img_size, augment=False, le_index=0, re_index=1, rotation_angles=[5,-5]):
    """
    Generate lmdb data.
    """
    data = getDataFromTxt(meta_txt, img_base_dir)
    imgs = []
    landmarks = []
    eyedists = []
    for i, (img_path, bbox, landmark) in enumerate(data):
        # read image 
        if is_color:
           channel = 3
           img = cv2.imread(img_path, 1)
        else:
           channel = 1
           img = cv2.imread(img_path, 0)
        assert(img is not None)
        log("process %s" % img_path)

        # enlarge face bounding box and crop, resize the face  
        enlarged_bbox = bbox.enlarge(0.05, 0.05, 0.05, 0.05)
        if not enlarged_bbox.valid(img.shape[0], img.shape[1]):
            enlarged_bbox = bbox
        face_original = img[int(enlarged_bbox.top):int(enlarged_bbox.bottom) + 1, int(enlarged_bbox.left):int(enlarged_bbox.right) + 1]
        face_original = cv2.resize(face_original, (img_size, img_size))
        # normalize landmark to range [0, 1]
        landmark_original = enlarged_bbox.normalize_landmarks(landmark)
        # put into container
        imgs.append(face_original.reshape((channel, img_size, img_size)))
        landmarks.append(landmark_original.reshape(10))
        eyedist = math.sqrt(
            (landmark_original[le_index][0] - landmark_original[re_index][0]) * (landmark_original[le_index][0] - landmark_original[re_index][0]) + 
            (landmark_original[le_index][1] - landmark_original[re_index][1]) * (landmark_original[le_index][1] - landmark_original[re_index][1])
            )
        eyedists.append(eyedist)
        
        ## for debug 
        if i < 4:
            bbox = BBox([0, img_size, 0, img_size])
            draw_landmark_in_cropped_face(face_original, bbox.denormalize_landmarks(landmark_original), os.path.join("visualization", "original_" + os.path.basename(img_path)))
            draw_landmark_in_original_image(img, enlarged_bbox, landmark, os.path.join("visualization", os.path.basename(img_path)))
        
        # data augmentation
        if augment:
            # horizontally flip 
            face_flipped, bbox_flipped, landmark_flipped = flip(img, enlarged_bbox, landmark)
            face_flipped = cv2.resize(face_flipped, (img_size, img_size))
            landmark_flipped = bbox_flipped.normalize_landmarks(landmark_flipped)
            
            imgs.append(face_flipped.reshape((channel, img_size, img_size)))
            landmarks.append(landmark_flipped.reshape(10))
            eyedists.append(eyedist)
            ## for debug
            if i < 4:
                bbox = BBox([0, img_size, 0, img_size])
                draw_landmark_in_cropped_face(face_flipped, bbox.denormalize_landmarks(landmark_flipped), os.path.join("visualization", "flipped_" + os.path.basename(img_path)))
            
            # rotate with probability 50%
            for alpha in rotation_angles:
                if np.random.rand() > 0.5:
                    img_rotated, face_rotated, landmark_rotated = rotate(img, enlarged_bbox, \
                        landmark, alpha)
                    # normalize the landmark to range [0,1]
                    landmark_rotated = enlarged_bbox.normalize_landmarks(landmark_rotated)
                    # resize face 
                    face_rotated = cv2.resize(face_rotated, (img_size, img_size))
                    imgs.append(face_rotated.reshape((channel, img_size, img_size)))
                    landmarks.append(landmark_rotated.reshape(10))
                    eyedists.append(eyedist)
                    ## for debug
                    if i < 4:
                        bbox = BBox([0, img_size, 0, img_size])
                        draw_landmark_in_cropped_face(face_rotated, bbox.denormalize_landmarks(landmark_rotated), os.path.join("visualization", "rotated_" + str(alpha) + "_" + os.path.basename(img_path)))
                
                    # horizontally flip the rotated face
                
                    face_flipped, bbox_flipped, landmark_flipped = flip(img_rotated, enlarged_bbox, enlarged_bbox.denormalize_landmarks(landmark_rotated))
                    face_flipped = cv2.resize(face_flipped, (img_size, img_size))
                    landmark_flipped = bbox_flipped.normalize_landmarks(landmark_flipped)
                    imgs.append(face_flipped.reshape((channel, img_size, img_size)))
                    landmarks.append(landmark_flipped.reshape(10))
                    eyedists.append(eyedist)

                    ## for debug
                    if i < 4:
                        bbox = BBox([0, img_size, 0, img_size])
                        draw_landmark_in_cropped_face(face_flipped, bbox.denormalize_landmarks(landmark_flipped), os.path.join("visualization", "flipped_rotated_" + str(alpha) + "_" + os.path.basename(img_path)))
                
            
    assert(len(imgs) == len(landmarks) and len(imgs) == len(eyedists))
    log('number of total generated images: %s' % str(len(imgs)))
        
    imgs, landmarks, eyedists = np.asarray(imgs), np.asarray(landmarks), np.asarray(eyedists)
    
    
    # shuffle the imgs, landmarks and eyedists
    rng_state = np.random.get_state()
    np.random.shuffle(imgs)
    np.random.set_state(rng_state)
    np.random.shuffle(landmarks)
    np.random.set_state(rng_state)
    np.random.shuffle(eyedists)    

    # create base dir if not existed
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output = os.path.join(output_dir, lmdb_name)
    log("generate %s" % output)

    # writw image to lmdb    
    in_db = lmdb.open(output + '_data', map_size=1e12)  
    with in_db.begin(write=True) as in_txn:  
        for in_idx, im in enumerate(imgs):  
            im = im[::-1,:,:]  
            im_dat = caffe.io.array_to_datum(im)  
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())  
    in_db.close() 
    
    # compute image mean 
    if augment:
        cmd = "../../build/tools/compute_image_mean " + output + "_data " + output_dir + "/train_mean.binaryproto"
    else:
        cmd = "../../build/tools/compute_image_mean " + output + "_data " + output_dir + "/test_mean.binaryproto"
    os.system(cmd)

    # write landmarks to lmdb
    in_db = lmdb.open(output + '_landmark', map_size=1e12)  
    count = 0  
    with in_db.begin(write=True) as in_txn:  
        for landmark in landmarks:  
            datum = caffe.proto.caffe_pb2.Datum()  
            datum.channels = landmark.shape[0]
            datum.height = 1
            datum.width = 1
            datum.float_data.extend(landmark.astype(float).flat)
            in_txn.put("{:0>10d}".format(count), datum.SerializeToString())  
            count += 1  
    in_db.close()  

    # write eyedists to lmdb
    in_db = lmdb.open(output + '_eyedist', map_size=1e12)  
    count = 0  
    with in_db.begin(write=True) as in_txn :  
        for eyedist in eyedists:  
            datum = caffe.proto.caffe_pb2.Datum()  
            datum.channels = eyedists.shape[0]
            datum.height = 1
            datum.width = 1
            datum.float_data.extend(eyedist.astype(float).flat)
            in_txn.put("{:0>10d}".format(count), datum.SerializeToString())  
            count += 1  
    in_db.close()  
    


if __name__ == '__main__':
    # train data
    # TRAIN = 'dataset/train'
    # assert(len(sys.argv) == 4)
    # isColor = int(sys.argv[1])
    # img_size = int(sys.argv[2])
    # OUTPUT = sys.argv[3]
    # if not exists(OUTPUT): os.mkdir(OUTPUT)
    # assert(exists(TRAIN) and exists(OUTPUT))
    train_txt = os.path.join('/share/disk/zengjiajian/AFW', 'label.txt')
    generate_lmdb(train_txt, '/share/disk/zengjiajian/AFW', 'dataset/train', 'train_afw_lmdb', True, 224, augment=True)

    #test_txt = join(TRAIN, 'testImageList.txt')
    #generate_lmdb(test_txt, OUTPUT, 'test_lmdb', isColor, img_size)

  
    # Done
