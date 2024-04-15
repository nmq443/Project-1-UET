import streamlit as st
from PIL import Image
from ultralytics import YOLO
import settings
import cv2
import helper

model_path = '../model/yolov8n.pt'

# Title of main page
st.title('Object Detection and Tracking using YOLOv8')

# Side bar
with st.sidebar: 
    model_task = st.radio(
        'Choose a task',
        ('Detection', 'Segmentation')
    )

    file_types = st.radio(
        'Perform on image or video?',
        ('Image:receipt:', 'Video:movie_camera:')
    )

    model_confidence_threshold = st.slider(
        "Model's confidence threshold?",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1,
        format='%.2f'
    ) / 100

    detect = st.button(
        label='Perform task',
        use_container_width=True
    )

# Input data
source_img, source_video = None, None

# Check file type?
if file_types == 'Image:receipt:':
    # Image uploader
    source_img = st.file_uploader(
        label='Upload your image here: ',
        type=['png', 'jpg'],
    )
elif file_types == 'Video:movie_camera:':
    # Video uploader
    source_video = st.file_uploader(
        label='Upload your video here: ',
    )


# Load model
if model_task == 'Detection':
    model_path = settings.DETECTION
elif model_task == 'Segmentation':
    model_path = settings.SEGMENTATION

model = YOLO(model_path)

if detect:
    if source_img != settings.DEFAULT_IMAGE and source_img != None:
        uploaded_img = Image.open(source_img)
        results = model.predict(uploaded_img, conf=model_confidence_threshold)

        # Split page into 2 columns
        col1, col2 = st.columns(2)

        # Display raw image on col1
        with col1:
            st.image(
                image=uploaded_img,
                caption='Raw image'
            )

        # Display prediction image on col2
        with col2:
            for result in results:
                res_bgr = result.plot()
                res_rgb = Image.fromarray(res_bgr[..., ::-1] )
                st.image(
                    image=res_rgb,
                    caption='Predicted image'
                )
    elif source_video != settings.DEFAULT_VIDEO:
        uploaded_video = helper.save_uploaded_file(source_video)
        uploaded_video = cv2.VideoCapture(uploaded_video)

        success = True
        while success:
            success, frame = uploaded_video.read()

            results = model.track(frame, conf=model_confidence_threshold)

            for result in results:
                res_bgr = result.plot()
                res_rgb = Image.fromarray(res_bgr[..., ::-1] )
                st.image(
                    image=res_rgb,
                )

