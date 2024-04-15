import torch

# Root directory
ROOT = '.'

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default image 
DEFAULT_IMAGE = '../test/dog.jpg'
DEFAULT_VIDEO = '../test/test_video.mp4'

# Model task
DETECTION = '../model/best.pt'
SEGMENTATION = '../model/yolov8n-seg.pt'
