import os.path
import time
import cv2
import numpy as np

def log(msg):
    """
    Log a message to the stdout.
    """
    now = time.ctime()
    print("[%s] %s" % (now, msg))

def draw_landmark_in_original_image(img, bbox, landmark, file_path):
    """
    Draw landmark in original image and write it to file.
    param:
    -img, of shape (h * w * c) and BGR format
    -bbox, the bounding box to draw, absolute coordinates
    -landmark, the landmarks to draw, of shape (N, 2) and absolute coordinates
    """
    img_copied = img.copy().astype(np.uint8)
    cv2.rectangle(img_copied, (int(bbox.left), int(bbox.top)), (int(bbox.right), int(bbox.bottom)), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img_copied, (int(x), int(y)), 2, (0,255,0), -1)
    cv2.imwrite(file_path, img_copied)
    

def draw_landmark_in_cropped_face(face, landmark, file_path):
    """
    Draw landmark in cropped face and write it to file.
    param:
    -face, of shape (h * w * c) and BRG format
    -landmark, using absolute position
    """
    face_copied = face.copy().astype(np.uint8)
    for (x, y) in landmark:
        cv2.circle(face_copied, (int(x), int(y)), 2, (0,255,0), -1)
    cv2.imwrite(file_path, face_copied)

def getDataFromTxt(txt, img_base_dir, with_landmark=True, num_landmarks=5):
    """
    Generate data from txt file. Each line in the text file is of format (img_name, x1, x2, y1, y2, lm1_x, lm1_y, lm2_x, lm2_y, ...).
 
    return: [(img_path, bbox, landmark)]
    -img_path: full file path of the image
    -bbox: [left, right, top, bottom]
    -landmark: [(x1, y1), (x2, y2), ...]
    """
    result = []
    with open(txt, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            line = line.strip()
            contents = line.split(' ')
            # full image path
            img_path = os.path.join(img_base_dir, contents[0].replace('\\', '/')) 
            # bounding box of format (left, right, top, bottom)
            bbox = (contents[1], contents[2], contents[3], contents[4])
            bbox = [float(_) for _ in bbox]
            bbox = BBox(bbox)
            if not with_landmark:
                result.append((img_path, bbox))
                continue
            # landmark 
            landmark = np.zeros((num_landmarks, 2))
            for index in range(0, num_landmarks):
                lm = (float(contents[5 + 2 * index]), float(contents[5 + 2 * index + 1]))
                landmark[index] = lm
            result.append((img_path, bbox, landmark))
    return result

def getDataFromTxtAFLW(txt, img_base_dir, with_landmark=True, num_landmarks=5):
    """
    Generate data from txt file. Each line in the text file is of format (img_name, x1, x2, y1, y2, lm1_x, lm1_y, lm2_x, lm2_y,...,v1, v2,...,yaw).
 
    return: [(img_path, bbox, landmark, visibility)]
    -img_path: full file path of the image
    -bbox: [left, right, top, bottom]
    -landmark: [(x1, y1), (x2, y2), ...]
    -visibility: [v1, v2, ...]
    """
    result = []
    with open(txt, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            line = line.strip()
            contents = line.split(' ')
            # full image path
            img_path = os.path.join(img_base_dir, contents[0].replace('\\', '/')) 
            # bounding box of format (left, right, top, bottom)
            bbox = (contents[1], contents[2], contents[3], contents[4])
            bbox = [float(_) for _ in bbox]
            bbox = BBox(bbox)
            if not with_landmark:
                result.append((img_path, bbox))
                continue
            # landmark 
            landmark = np.zeros((num_landmarks, 2))
            for index in range(0, num_landmarks):
                lm = (float(contents[5 + 2 * index]), float(contents[5 + 2 * index + 1]))
                landmark[index] = lm
            # visibility
            visibility = np.zeros(num_landmarks, dtype=np.int)
            for index in range(0, num_landmarks):
                v = int(contents[5 + 2 * num_landmarks + index])
                visibility[index] = v
            result.append((img_path, bbox, landmark, visibility))
    return result

class BBox(object):
    """
    Bounding box of face which comprises a four-elements tuple (x, y, w, h).
    (x, y) is the top left coordinate of the box, and w, h is the width and height of the box respectively.
    """
    def __init__(self, bbox, lrtb=True):
        # bbox is of format (x1, x2, y1, y2)
        if lrtb:
            self.left = bbox[0]
            self.right = bbox[1]
            self.top = bbox[2]
            self.bottom = bbox[3]
            
        # bbox if of format (x1, y1, x2, y2)
        else:
            self.left = bbox[0]
            self.right = bbox[2]
            self.top = bbox[1]
            self.bottom = bbox[3]
        self.x = self.left
        self.y = self.top
        self.w = self.right - self.left
        self.h = self.bottom - self.top

    # normalize a point coordinate within this box to range [0, 1]
    def normalize(self, point):
        x = float((point[0] - self.x)) / self.w
        y = float((point[1] - self.y)) / self.h
        return np.asarray([x, y])
    # denormalize a point coordinate in range [0, 1] back
    def denormalize(self, point):
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    # normalize a set of landmarks belonging to this box
    def normalize_landmarks(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.normalize(landmark[i])
        return p
    # denormalize a set of landmarks belonging to this box
    def denormalize_landmarks(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.denormalize(landmark[i])
        return p
    
    # enlarge this bounding box by a small ratio (leftR, rightR, topR, bottomR)
    # in general case, a appropriate ratio is (0.05, 0.05, 0.05, 0.05)
    def enlarge(self, leftR, rightR, topR, bottomR):
        left = self.left - self.w * leftR
        right = self.right + self.w * rightR
        top = self.top - self.h * topR
        bottom = self.bottom + self.h * bottomR
        
        return BBox([left, right, top, bottom])

    # in order to make sure that when the face bounding box is incorrectly labeld, the coordinates, width and height computation
    # is correct
    def misc_clip(self, height, width):
        if self.left < 0 or self.left > width:
            self.left = 0
        if self.right < 0 or self.right > width:
            self.right = width - 1
        if self.top < 0 or self.top > height:
            self.top = 0
        if self.bottom < 0 or self.bottom > height:
            self.bottom = height - 1
        self.x = self.left
        self.y = self.top
        self.w = self.right - self.left
        self.h = self.bottom - self.top
        
    # check is that bbox valid or not
    def valid(self, height, width):
        if self.left >= 0 and self.left <= width \
           and self.right >= 0 and self.right <= width \
           and self.top >=0 and self.top <= height \
           and self.bottom >= 0 and self.bottom <= height \
           and self.left < self.right and self.top < self.bottom:
           return True;
        return False;
