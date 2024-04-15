import streamlit as st
from PIL import Image
from ultralytics import YOLO
import settings
import cv2
import helper
import tempfile
import pytube
from pytube import YouTube

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

        st_frame = st.empty()

        # TODO: fix this bug, still can't play video
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                helper.display_single_frame(conf=model_confidence_threshold, model=model, st_frame=st_frame, frame=frame)

            else:
                cap.release()
                break

elif file_types == 'Webcam':
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        
        results = model.track(frame, conf=model_confidence_threshold, persist=True)
        frame_bgr = results[0].plot()
        frame_rgb = Image.fromarray(frame_bgr[..., ::-1])

        # visualization
        frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')
        print("Chose Webcam")

elif file_types == 'Youtube':
    print("Youtube")
    url = st.text_input('URL link')

    if url != '':
        preds = model.predict(source=url, stream=True)
        st.write(preds)
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension="mp4", res=720).first()
        vid_cap = cv2.VideoCapture(stream.url)

        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                helper.display_single_frame(conf=model_confidence_threshold,
                                         model=model,
                                         st_frame=st_frame,
                                         frame=image,
                                         )
            else:
                vid_cap.release()
                break
        """
        yt = YouTube(url)
        
        stream = yt.streams.filter(file_extension='mp4', res=720).first()
        cap = cv2.VideoCapture(stream.url)

        success = True

        st_frame = st.empty()

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                helper.display_single_frame(conf=model_confidence_threshold, model=model, st_frame=st_frame, frame=frame)

            else:
                cap.release()
                break

        """
