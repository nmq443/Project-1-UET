# import dependencies
import cv2
from ultralytics import YOLO

# init a model
model = YOLO('result/runs/detect/train/weights/best.pt')