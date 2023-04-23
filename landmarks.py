from retinaface import RetinaFace
import cv2
import mtcnn
import scrfd
# update aug 9 2021, now the functions should return both cropped face and landmarks, and landmarks have been converted to the cropped coordinate.

"""
In this project, the landmark array we use is as following:
landmark = [[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]],
                   [nose[0], nose[1]],
                   [mouth_left[0], mouth_left[1]],
                   [mouth_right[0], mouth_right[1]]]

due to different detect method have different numbers and format of landmarks, thus I rewrote this file in order it could support different detection methods.
"""

"""
faces = RetinaFace.detect_faces(img_path = 'test.jpg')
faces = {'face_1': {'score': 0.9989180564880371,
  'facial_area': [86, 76, 167, 192],
  'landmarks': {'right_eye': [109.999374, 118.28145],
   'left_eye': [146.62201, 117.99363],
   'nose': [129.83868, 135.41493],
   'mouth_right': [111.90126, 157.98032],
   'mouth_left': [146.38423, 157.41806]}}}

"""
def retina(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = RetinaFace.detect_faces(img)
    print(faces)
    landmarks = []
    cropfaces = []
    if(not type(faces) == dict):
            raise Exception("No face detected")
    for facekey in faces.keys():
        #solves faces
        facearea = faces[facekey]['facial_area']
        x0,y0,x1,y1 = facearea[0],facearea[1],facearea[2],facearea[3]
        face = img[y0:y1,x0:x1]
        cropfaces.append(face)

        # solves landmarks
        lmks = faces[facekey]['landmarks']
        right_eye = lmks['left_eye']
        left_eye = lmks['right_eye']
        nose = lmks['nose']
        #notice that the keypoints in retinaface is oppsite to the view
        mouth_right = lmks['mouth_left']
        mouth_left = lmks['mouth_right']
        landmark = [[left_eye[0], left_eye[1]],
                    [right_eye[0], right_eye[1]],
                    [nose[0], nose[1]],
                    [mouth_left[0], mouth_left[1]],
                    [mouth_right[0], mouth_right[1]]]
        # transform to cropped corridnator
        for i in range(len(landmark)):
            landmark[i][0] -= x0
            landmark[i][1] -= y0
        landmarks.append(landmark)
    return cropfaces,landmarks



def mtcnn_landmarks(img):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = mtcnn.MTCNN()
        faces = detector.detect_faces(img)
        crop_faces = []
        landmarks = []
        if(len(faces) == 0):
            raise Exception("No face detected")
        for face in faces:
                x0,y0,w,h = face['box'][0],face['box'][1],face['box'][2],face['box'][3]
                left_eye = face["keypoints"]["left_eye"]
                right_eye = face["keypoints"]["right_eye"]
                nose = face["keypoints"]["nose"]
                mouth_left = face["keypoints"]["mouth_left"]
                mouth_right = face["keypoints"]["mouth_right"]
                landmark = [[left_eye[0], left_eye[1]],
                                [right_eye[0], right_eye[1]],
                                [nose[0], nose[1]],
                                [mouth_left[0], mouth_left[1]],
                                [mouth_right[0], mouth_right[1]]]
                for i in range(len(landmark)):
                        landmark[i][0] -= x0
                        landmark[i][1] -= y0
                crop = img[y0:y0+h,x0:x0+w]
                landmarks.append(landmark)
                crop_faces.append(crop)
        return crop_faces,landmarks




def scrfd_landmarks(img):
    crop_faces,landmarks = scrfd.detect(img)
    return crop_faces,landmarks


def get_landmarks(img,model='retina'):
    if(model == 'retina'):
        return retina(img)
    elif(model == 'scrfd'):
        return scrfd_landmarks(img)
    elif(model == 'mtcnn'):
        return mtcnn_landmarks(img)
    else:
        print('no such detect method')
