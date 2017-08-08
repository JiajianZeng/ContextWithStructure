import argparse
import lmdb
import caffe
import PIL.Image
import numpy as np
import os.path
import sys
from util import draw_landmark_in_cropped_face, BBox

def read_image_from_lmdb(lmdb_file, num_to_read, save_dir, vis = True, mode = "RGB", ext = ".jpg", print_info=True):
    """
    Read images from lmdb file.
    
    return: image array
    """
    lmdb_env = lmdb.open(lmdb_file, readonly=True)
    lmdb_txn = lmdb_env.begin()
    cursor = lmdb_txn.cursor()
    imgs = []
    # print some stat info
    if print_info:
        print '##### lmdb %s #####' % os.path.basename(lmdb_file)
        print '##### lmdb stat #####'
        print lmdb_env.stat()
        print '##### end lmdb stat #####'
    datum = caffe.proto.caffe_pb2.Datum()
    i = 0
    for _, value in cursor:
        if i >= num_to_read:
            break;
        datum.ParseFromString(value)
        if i == 0 and print_info:
            print '##### channels, width, height #####'
            print datum.channels, datum.width, datum.height
            print '##### end channels, width, height #####'
            print '##### lmdb %s #####' % os.path.basename(lmdb_file)
        im_array = caffe.io.datum_to_array(datum)
        # for visualization purpose
        if vis:
            # from c x h x w -> h x w x c
            im_array = np.transpose(im_array, (1,2,0))       
            img = PIL.Image.fromarray(im_array, mode)
            img.save(os.path.join(save_dir, str(i) + ext))
            # RGB -> BGR
            imgs.append(im_array[:, :, ::-1])
        # for readonly purpose
        else:
            imgs.append(im_array)
        i += 1
    lmdb_env.close()
    return imgs

def read_label_from_lmdb(lmdb_file, num_to_read, print_info=True):
    """
    Read labels from lmdb file.
    """
    lmdb_env = lmdb.open(lmdb_file, readonly=True)
    lmdb_txn = lmdb_env.begin()
    cursor = lmdb_txn.cursor()
    labels = []
    # print stat info
    if print_info:
        print '##### lmdb %s #####' % os.path.basename(lmdb_file)
        print '##### lmdb stat #####'
        print lmdb_env.stat()
        print '##### end lmdb stat #####'
    datum = caffe.proto.caffe_pb2.Datum()
    i = 0
    for _, value in cursor:
        if i >= num_to_read:
            break;
        datum.ParseFromString(value)
        # print stat info
        if i == 0 and print_info:
            print '##### channels, width, height #####'
            print datum.channels, datum.width, datum.height
            print '##### end channels, width, height #####'
            print '##### lmdb %s #####' % os.path.basename(lmdb_file)
        label = caffe.io.datum_to_array(datum)
        labels.append(label)
        i += 1
    lmdb_env.close()
    return labels
    
def parse_args():
    parser = argparse.ArgumentParser(description='Read images or ground-truth labels from lmdb.')
    # image lmdb
    parser.add_argument('--lmdb_data',
                        help='image lmdb file',
                        default=None, type=str)
    # landmark lmdb
    parser.add_argument('--lmdb_landmark', 
                        help='landmark lmdb file', 
                        default=None, type=str)
    # number to read
    parser.add_argument('--num_to_read', 
                        help='number of images to read',
                        default=4, type=int)
    # visualization save dir
    parser.add_argument('--save_dir',
                        help='where to save the visualization result',
                        default='./image_read_from_lmdb', type=str)
    # number of landmarks to be visualized
    parser.add_argument('--num_landmarks',
                         help='number of landmarks to be visualized',
                         default=5, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # get arguments
    args = parse_args()
    lmdb_data = args.lmdb_data
    lmdb_landmark = args.lmdb_landmark
    num_to_read = args.num_to_read
    save_dir = args.save_dir
    num_landmarks = args.num_landmarks
    # read images and landmarks
    imgs = read_image_from_lmdb(lmdb_data, num_to_read, save_dir)
    landmarks = read_label_from_lmdb(lmdb_landmark, num_to_read)
    # visualization
    bbox = BBox([0, imgs[0].shape[1], 0, imgs[0].shape[0]])
    for i in range(num_to_read):
        draw_landmark_in_cropped_face(imgs[i], bbox.denormalize_landmarks(landmarks[i].reshape(num_landmarks, 2)), os.path.join(save_dir, "recovered_" + str(i) + '.jpg'))

    




