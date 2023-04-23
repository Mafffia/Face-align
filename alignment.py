"""This is the py file for face align, we try to use the align tech in ArcFace, rather than the align in DeepFace"""
import cv2
import mtcnn
import numpy as np
from skimage import transform as trans
import cv2
from PIL import Image

import math

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def align_rotate(img,landmarks):
    left_eye_center = (landmarks[0][0],landmarks[0][1])
    right_eye_center = (landmarks[1][0],landmarks[1][1])
    left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
    right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        # print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        # print("rotate to inverse clock direction")
    # cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)

    # cv2.line(img,right_eye_center, left_eye_center,(67,67,67),2)
    # cv2.line(img,left_eye_center, point_3rd,(67,67,67),2)
    # cv2.line(img,right_eye_center, point_3rd,(67,67,67),2)

    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, left_eye_center)
    c = euclidean_distance(right_eye_center, point_3rd)
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = np.arccos(cos_a)
    angle = (angle * 180) / math.pi
    if direction == -1:
        angle = 90 - angle
    new_img = Image.fromarray(img)
    new_img = np.array(new_img.rotate(direction * angle))
    return new_img



arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

# arcface_src = np.expand_dims(arcface_src, axis=0)


# project the face image to the nearest template face
# in the paper only ArcFace src is used: there is only one face template, and there is no comparison
# see https://github.com/deepinsight/insightface/blob/e3db188ce8a376f5b1df07d05e502e02c94be118/python-package/insightface/utils/face_align.py

def estimate_norm(lmk, image_size=224,method='affine'):
    assert lmk.shape == (5, 2)
    if(method == 'affine'):
        tform = trans.AffineTransform()
    elif(method == 'similarity'):
        tform = trans.SimilarityTransform()
    ratio = float(image_size)/112.0
    dst = arcface_src * ratio
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=224,method='affine'):
    M = estimate_norm(landmark, image_size,method)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


""""Input: cropped face
    Output: aligned face
    landmark = [[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]],
                   [nose[0], nose[1]],
                   [mouth_left[0], mouth_left[1]],
                   [mouth_right[0], mouth_right[1]]]
"""

    
     
def get_face_align(cropface,landmark,method='affine'):
    if(method=='rotate'):
        return align_rotate(cropface,landmark)
    img = cv2.cvtColor(cropface, cv2.COLOR_BGR2RGB)  # To RGB
    
    # landmark = detect_landmark(img, detector)
    wrap = norm_crop(img, np.array(landmark), image_size=224,method=method)
    wrap = cv2.cvtColor(wrap,cv2.COLOR_BGR2RGB)
    return wrap
