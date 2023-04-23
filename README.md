# Face-align
Providing three face alignment methods : Affine, similarity transformation, and rotation


## Dependency
+ [DeepFace](https://github.com/serengil/deepface)
+ [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)
+ [opencv-python](https://pypi.org/project/opencv-python/)
+ [scikit-image](https://scikit-image.org/docs/stable/install.html)
+ [mtcnn](https://pypi.org/project/mtcnn/)
+ SCRFD(provided in this repo, fetched from [InsightFace](https://github.com/deepinsight/insightface))
+ [onnx](https://pypi.org/project/onnx/) and [onnx-runtime](https://onnxruntime.ai/docs/install/) (for SCRFD support)


## Install dependency

    pip install numpy==1.23.2 deepface==0.0.75 Pillow==9.2.0 scikit-image==0.19.3 mtcnn==0.1.1 tensorflow==2.9.1 onnx==1.12.0 onnxruntime==1.12.0

## Usage in DeepFace
by default, the deepface uses detectors to detect faces, but since when performing alignment, the face will be cropped first,
there is no need to make another detection in DeepFace. To disable detector in DeepFace, sepcify 

    detector_backend = 'skip'
See example.ipynb on useage

