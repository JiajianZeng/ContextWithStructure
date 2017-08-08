import os
import cv2
import numpy as np
from util import BBox

def flip(img, bbox, landmark):
    """
    Flip image horizontally as well as the bbox and landmark.
    
    param:
    -img, the original image
    -landmark, using absolute coordinates
    """
    # horizontally flip the image
    img_flipped = cv2.flip(img.copy(), 1)
    # horizontally flip the bbox and crop the bbox region
    bbox_flipped = BBox([img.shape[1] - bbox.right, img.shape[1] - bbox.left, bbox.top, bbox.bottom])
    img_flipped = img_flipped[int(bbox_flipped.top):int(bbox_flipped.bottom) + 1, int(bbox_flipped.left):int(bbox_flipped.right) + 1]    
    # horizontally flip the landmark
    landmark_flipped = np.asarray([(img.shape[1]-x, y) for (x, y) in landmark])
    
    # exchange the right eye and left eye, right mouth corner and left mouth corner
    # however, this may not be needed
    landmark_flipped[[0, 1]] = landmark_flipped[[1, 0]]
    landmark_flipped[[3, 4]] = landmark_flipped[[4, 3]]
    return (img_flipped, bbox_flipped, landmark_flipped)

def rotate(img, bbox, landmark, alpha):
    """
    Given an image with bbox and landmark, rotate it with (angle in degrees) alpha and 
    return rotated face with bbox and landmark (absolute coordinates)

    param:
    -img, the original image
    -landmark, using absolute coordinates
    -alpha, rotation angle in degrees
    """
    # get rotation matrix
    center = (float(bbox.left + bbox.right) / 2, float(bbox.top + bbox.bottom) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    # apply rotation, (img.shape[1], img.shape[0]) means (width, height)
    img_rotated = cv2.warpAffine(img.copy(), rot_mat, (img.shape[1], img.shape[0]))
    # calculate new landmark
    landmark_rotated = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                 rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])  
    face_cropped = img_rotated[int(bbox.top):int(bbox.bottom) + 1, int(bbox.left):int(bbox.right) + 1]
    return (img_rotated, face_cropped, landmark_rotated)
