import streamlit as st
from PIL import Image
from ultralytics import YOLO
import settings
import cv2
import helper
import tempfile

model_path = '../model/yolov8n.pt'

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

    detect = st.button(
        label='Perform task',
        use_container_width=True
    )


# Input data
source_img, source_video = None, None

# Load model
if model_task == 'Detection':
    model_path = settings.DETECTION
elif model_task == 'Segmentation':
    model_path = settings.SEGMENTATION

model = YOLO(model_path)

# Check source's file type?
if file_types == 'Image':
    # Image uploader
    source_img = st.sidebar.file_uploader(
        label='Upload your image here: ',
        type=['png', 'jpg'],
    )

    if source_img == None:
        default_img = settings.DEFAULT_IMAGE
        results = model.predict(default_img, conf=model_confidence_threshold)

        # Split page into 2 cols
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image=default_img,
                caption='Raw image'
            )

        with col2:
            helper.image_object_detection(conf=model_confidence_threshold, image=default_img, model=model)
    else:
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
        if detect:
            # Display prediction image on col2
            with col2:
                helper.image_object_detection(conf=model_confidence_threshold, image=uploaded_img, model=model)

elif file_types == 'Video':
    # Video uploader
    source_video = st.sidebar.file_uploader(
        label='Upload your video here: ',
    )

    if source_video != None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(source_video.read())
        cap = cv2.VideoCapture(temp_file.name)

        success = True

        # get vcap property 
        width  = int(cap.get(3))   # float `width`
        height = int(cap.get(4))  # float `height`

        fps = int(cap.get(5))

        video = cv2.VideoWriter('predicted_video.mp4', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

        # TODO: fix this bug, still can't play video
        while success:
            success, frame = cap.read()

            if success:
                results = model.track(frame, persist=True)
                # plot result
                frame_ = results[0].plot()

                # visualization
                # cv2.imshow('frame', frame_)
                video.write(frame_)

                if cv2.waitKey(25):
                    break
        video.release()

        st.video(video)


    
elif file_types == 'Webcam':
    print("Chose Webcam")

elif file_types == 'Youtube':
    print("Chose Youtube")


