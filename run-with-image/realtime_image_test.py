# import dependencies
import cv2
from ultralytics import YOLO

# load model
#model = YOLO('./results/runs/detect/train/weights/best.pt')
model = YOLO('../model/yolov8n.pt')

# load image 
image_path = '../test/test_image.png'

result = model(image_path)
cv2.imshow('frame', result[0].plot())
cv2.waitKey(5000)

"""
# read frames
ret = True

while ret:
    ret, frame = cap.read()
    # cv2.imshow('yolov8', frame)
    
    if ret:
        # tracking and detecting objects
        results = model.track(frame, persist=True)

        # plot result
        frame_ = results[0].plot()

        # visualization
        cv2.imshow('frame', frame_)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

"""