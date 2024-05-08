import streamlit as st
from PIL import Image
from ultralytics import YOLO
import settings
import cv2
import helper
import tempfile
import pytube
from pytube import YouTube

model_path = '../model/vehicles.pt'

# Title of main page
st.title('Object Detection and Tracking using YOLOv8')

# Side bar
with st.sidebar: 
    st.header("ML Model config")
    # Task selection
    model_task = st.radio(
        'Choose a task',
        ('Detection', 'Segmentation')
    )

    st.header("Image/Video data")
    # Source file type
    file_types = st.radio(
        'Select source?',
        settings.SOURCES
    )

    model_confidence_threshold = st.slider(
        "Model's confidence threshold?",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1,
        format='%.2f'
    ) / 100

    perform_task_button = st.button(
        label='Perform task',
        use_container_width=True
    )


# Input data
source_img, source_video = None, None

# Setting detection OR segmentation task 
if model_task == 'Detection':
    model_path = settings.DETECTION
elif model_task == 'Segmentation':
    model_path = settings.SEGMENTATION

# Load model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Check source's file type?
if file_types == 'Image':
    # Image uploader
    source_img = st.sidebar.file_uploader(
        label='Upload your image here: ',
        type=['png', 'jpg'],
    )
    # Split page into 2 cols
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img == None:
                default_img = settings.DEFAULT_IMAGE
                results = model.predict(default_img, conf=model_confidence_threshold)
                
                st.image(
                    image=default_img,
                    caption='Raw image'
                )
            else:
                uploaded_img = Image.open(source_img)
                results = model.predict(uploaded_img, conf=model_confidence_threshold)
                st.image(
                    image=uploaded_img,
                    caption='Raw image'
                )
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img == None:
            helper.image_object_detection(conf=model_confidence_threshold, image=default_img, model=model)

        else:
            if perform_task_button:
                try:
                    helper.image_object_detection(conf=model_confidence_threshold, image=uploaded_img, model=model)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif file_types == 'Video':
    # Video uploader
    source_video = st.sidebar.file_uploader(
        label='Upload your video here: ',
    )
    is_tracking, tracker_type = helper.display_tracking_options()

    if source_video != None:
        if perform_task_button:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(source_video.read())
            video_src = temp_file.name

            helper.realtime_object_detection(
                    conf=model_confidence_threshold,
                    model=model,
                    video_src=video_src
            )

elif file_types == 'Webcam':
    st.title("Webcam Live Feed")
    helper.webcam_object_detection(
        conf=model_confidence_threshold,
        model=model,
    )

elif file_types == 'Youtube':
    url = st.text_input('URL link')
    if perform_task_button:
        helper.youtube_video_object_detection(
            conf=model_confidence_threshold,
            model=model,
            url=url
        )
