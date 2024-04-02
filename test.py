# import dependencies
import cv2
from ultralytics import YOLO

# load model
model = YOLO('./results/runs/detect/train/weights/best.pt')

# load video
video_path = 'Q:/Personal/Study Resources/Engineering/Lab 1/Project-1-UET/test/test_video.mp4'
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

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
