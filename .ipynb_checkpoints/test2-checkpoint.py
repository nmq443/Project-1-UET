from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO('./results/runs/detect/train/weights/best.pt')

image_path = 'Q:/Personal/Study Resources/Engineering/Lab 1/Project-1-UET/dataset/Vehicles-coco.v2i.yolov8/test/images/000000019193_jpg.rf.c7988d384d239a705692bc3f42c5c9f1.jpg'

res = model.predict(image_path)

