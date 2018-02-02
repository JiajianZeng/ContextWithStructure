# ContextWithStructure
*ContextWithStructure* is a deep learning based facial landmark detection framework, which jointly models the context as well as the intrinsic geometric structure of facial landmarks. This project hosts the full source code for our paper *Deep Context-Sensitive Facial Landmark Detection with Tree-Structured Modeling* (Accepted as a regular paper for *IEEE Transactions on Image Processing*, 2017). The early accessed version of the paper can be found [here](http://ieeexplore.ieee.org/document/8219746/).

In the following, we will provide the details of the **datasets** as well as the **instructions** to train and evaluate the models.

# Datasets
## <a name="download"></a>Download
Currently, we provide four datasets, which are [AFLW_FULL](https://drive.google.com/open?id=1KntIVs2VfhJb3T2zj36Iptqwi-csoEMi), [LFW_NET](https://drive.google.com/open?id=1WJ1ZxJsj4hhIshYRKdVJnNCYQMZodPwe), [MTFL_TEST](https://drive.google.com/open?id=195YLR6aUVcmZiW8kFTk6g18sZsD1qdUk) and [UMDFaces](https://drive.google.com/open?id=1aB-lVsBvjIIlD4sTLrRH3sVLpB_GJUjV). For *AFLW_FULL*, *LFW_NET* and *MTFL_TEST* datasets, both the images and annotations can be downloaded via the Google Drive link. As for the *UMDFaces* dataset, only the annotations are provided because of the storage limit of my Google Drive account (15GB vs 60GB). You can download it from its [official project page](http://www.umdfaces.io/). 

## Annotation
The annotations of these four datasets we provide are a bit different from their official versions. For example, the original annotations of the *MTFL_TEST* dataset include (1) five facial landmarks and (2) attributes of gender, smiling, wearing glasses and head pose. While our annotations (1) remove the four attributes and (2) offer the face bounding box. What we can guarantee is that **the annotations of all facial landmarks we supply are identical to the official versions**. 

The detailed descriptions of the annotations of each dataset are as follows:
* AFLW_FULL
  * img_path bbox_x1 bbox_x2 bbox_y1 bbox_y2 lm1_x lm1_y lm2_x lm2_y ... lm19_x lm19_y v1 v2 ... v19 img_h img_w
* LFW_NET
  * img_path bbox_x1 bbox_x2 bbox_y1 bbox_y2 lm1_x lm1_y lm2_x lm2_y ... lm5_x lm5_y
* MTFL_TEST
  * img_path bbox_x1 bbox_x2 bbox_y1 bbox_y2 lm1_x lm1_y lm2_x lm2_y ... lm5_x lm5_y
* UMDFaces
  * img_path bbox_x1 bbox_x2 bbox_y1 bbox_y2 lm1_x lm1_y lm2_x lm2_y ... lm19_x lm19_y
  
Where *img_path* means the file path of a image relative to the root directory of the dataset, *(bbox_x1, bbox_y1)* and *(bbox_x2, bbox_y2)* are the coordinates of the left top and right bottom points of the face bounding box respectively. *(lm#_x, lm#_y)* represents the coordinate of the #-th facial landmark, and *v#* indicates its visibility. 

## Processing
The data processing consists of two parts: 1) data augmentation and 2) preprocessing. Specifically, the data augmentation includes horizontal flip and rotation. And the preprocessing includes mean image subtraction and ground-truth landmark normalization. 

### Data augmentations

| Dataset | augmentation | # of training images | # of training images after augmentation |
| ------- | :----------: | :------------------: | :-------------------------------------: |
| AFLW_FULL | flip + rotation (-15 and +15 degrees)                                              | 20000 | 60000 |
|  LFW_NET  | flip + rotation (-15 and +15 degrees) + rotated flip (-15 and +15 degrees)           | 10000 | 60000 | 
| MTFL_TEST |                               x                                                  |   x   |   x   |
| UMDFaces  |                        no augmentation                                           | 317918| 317918|       

Here *rotated flip* means the horizontal flip after rotation, and the *MTFL_TEST* dataset is used as the test set only. 

### Preprocessings
1. Mean image subtraction which subtracts the mean image of the whole training set from each training image.
2. Ground-truth landmark normalization which normalizes the coordinate *(x, y)* of a specific landmark to the range 0 < x < 1, 0 < y < 1.

# Instructions
## Installation
- git clone https://github.com/JiajianZeng/ContextWithStructure.git
- cd $CWS_HOME
- refer [Caffe Installation](http://caffe.berkeleyvision.org/installation.html) to install

Here *$CWS_HOME* means the root directory of the cloned project.
## Training process
- download the datasets via the Google Drive link provided in the [Download](#download) section.
### Generate Training LMDB
- cd $CWS_HOME/experiments/data_processing
- AFLW_FULL
  - python generate_lmdb.py --meta_file $TRAIN_ANNO_FILE --img_base_dir $IMG_BASE_DIR --output_dir dataset/train/ --lmdb_prefix aflw_full_224x224_rgb --is_color True --img_size 224 --flipping false --rotation true --rotated_flipping false --num_landmarks 19 --rotation_angles 15 --rotation_angles -15
- LFW_NET
  - python generate_lmdb.py --meta_file $TRAIN_ANNO_FILE --img_base_dir $IMG_BASE_DIR --output_dir dataset/train/ --lmdb_prefix lfw_net_224x224_rgb --is_color True --img_size 224 --flipping true --rotation true --rotated_flipping true --num_landmarks 5 --rotation_angles 15 --rotation_angles -15
- UMDFaces
  - python generate_lmdb.py --meta_file $TRAIN_ANNO_FILE --img_base_dir $IMG_BASE_DIR --output_dir dataset/train/ --lmdb_prefix umd_face_224x224_rgb --is_color true --img_size 224 --flipping false --rotation false --rotated_flipping false --num_landmarks 19

Here *$TRAIN_ANNO_FILE* and *$IMG_BASE_DIR* represent the training annotation file and root directory to the images of the corresponding dataset respectively. When generating lmdb data, you can find some visualization results under the *visualization/* folder. And after the processing done, you will find the generated lmdb file under the *dataset/train/* folder. 

Besides, we also provide a tool to check whether the generated lmdb data is correct or not, for example, the following command
- python lmdb_util.py --lmdb_data dataset/train/umd_224x224_rgb_data/ --lmdb_landmark dataset/train/umd_224x224_rgb_landmark/ --num_to_read 20 --save_dir image_read_from_lmdb/ --num_landmarks 19

will recover the first 20 images from the lmdb data, the recovered images can be found under the *image_read_from_lmdb/* folder.
### Generate Test LMDB
- cd $CWS_HOME/experiments/data_processing
- AFLW_FULL
  - python generate_lmdb.py --test_stage True --meta_file $TEST_ANNO_FILE --img_base_dir $IMG_BASE_DIR --output_dir dataset/test/ --lmdb_prefix aflw_full_224x224_rgb --is_color True --img_size 224 --num_landmarks 19
- LFW_NET
  - python generate_lmdb.py --test_stage True --meta_file $TEST_ANNO_FILE --img_base_dir $IMG_BASE_DIR --output_dir dataset/test/ --lmdb_prefix lfw_net_224x224_rgb --is_color True --img_size 224 --num_landmarks 5
- MTFL_TEST
  - python generate_lmdb.py --test_stage True --meta_file $TEST_ANNO_FILE --img_base_dir $IMG_BASE_DIR --output_dir dataset/test/ --lmdb_prefix mtfl_test_224x224_rgb --is_color True --img_size 224 --num_landmarks 5
- UMDFaces
  - python generate_lmdb.py --test_stage True --meta_file $TEST_ANNO_FILE --img_base_dir $IMG_BASE_DIR --output_dir dataset/test/ --lmdb_prefix umd_face_224x224_rgb --is_color True --img_size 224 --num_landmarks 19

### Compute Mean Image 
- cd $CWS_HOME
- AFLW_FULL
  - ./build/tools/compute_image_mean ./experiments/data_processing/dataset/train/aflw_full_224x224_rgb_data ./experiments/data_processing/dataset/train/aflw_full_224x224_rgb_mean.binaryproto
- LFW_NET
  - ./build/tools/compute_image_mean ./experiments/data_processing/dataset/train/lfw_net_224x224_rgb_data ./experiments/data_processing/dataset/train/lfw_net_224x224_rgb_mean.binaryproto
- UMDFaces
  - ./build/tools/compute_image_mean ./experiments/data_processing/dataset/train/umd_face_224x224_rgb_data ./experiments/data_processing/dataset/train/umd_face_224x224_rgb_mean.binaryproto
