import torch

# Root directory
ROOT = '.'

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default image 
# DEFAULT_IMAGE = '../test/dog.jpg'
DEFAULT_IMAGE = './test/dog.jpg'
DEFAULT_VIDEO = './test/test_video.mp4'

# Model task
DETECTION = '../model/yolov8n.pt'
SEGMENTATION = '../model/yolov8n-seg.pt'

# Source file type
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
YOUTUBE = 'Youtube'

SOURCES = (IMAGE, VIDEO, WEBCAM, YOUTUBE)
