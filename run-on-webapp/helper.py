import streamlit as st
import os
import cv2
from PIL import Image
from pytube import YouTube

def display_tracking_options():
    """
    Tracking options, Yes if user want to display tracking on video, No o.w

    Args:
    None

    Returns:
    is_tracking (bool): tracking or not
    tracker_type (string): type of tracker
    """
    display_tracking = st.radio("Display tracker?", ('Yes', 'No'))
    is_tracking = display_tracking == 'Yes'
    if is_tracking:
        tracker_type = st.radio('Tracker', ('bytetrack.yaml', 'botsort.yaml'))
        return is_tracking, tracker_type
    return is_tracking, None 

def image_object_detection(image, conf, model):
    """
    Display detected object on an image

    Args:
    - image (numpy array): Image
    - conf (float): Confidence threshold
    - model (YOLO): YOLOv8 model

    Returns:
    None
    """
    results = model.predict(image, conf=conf)

    for result in results:
        res_bgr = result.plot()
        res_rgb = Image.fromarray(res_bgr[..., ::-1] )
        st.image(
            image=res_rgb,
            caption='Predicted image'
        )

def display_single_frame(model, conf, st_frame, frame, is_tracking=False, tracker=None): 
    """
    Display detected object on a single frame of the video.

    Args:
    - model (YOLO): A YOLOv8 model
    - con (float): Confidence threshold for object detection
    - st_frame (Streamlit object): A Streamlit object to display the detected video
    - frame (numpy.array): A numpy array representing the frame
    - is_tracking (bool): display tracking or not
    - tracker: type of tracker

    Returns:
    None
    """
    if is_tracking:
        results = model.track(frame, conf=conf, persist=True, tracker=tracker)
    else:
        results = model.predict(frame, conf=conf)
    frame_bgr = results[0].plot()
    frame_rgb = Image.fromarray(frame_bgr[..., ::-1])

    # visualization
    st_frame.image(frame_rgb)

def realtime_object_detection(video_src, conf, model):
    """
    Realtime object detection performs on video

    Args:
    - video_src (string): Path to video
    - conf (float): Confidence threshold
    - model (YOLO): YOLOv8 model

    Returns:
    None
    """
    cap = cv2.VideoCapture(video_src)

    success = True

    st_frame = st.empty()

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            display_single_frame(conf=conf, model=model, st_frame=st_frame, frame=frame)

        else:
            cap.release()

def webcam_object_detection(model, conf):
    """
    Realtime object detection performs on webcam 

    Args:
    - conf (float): Confidence threshold
    - model (YOLO): YOLOv8 model

    Returns:
    None
    """
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

def youtube_video_object_detection(model, conf, url):
    """
    Object detection on YouTube video
    Parameters:
        model (YOLO): YOLOv8 model
        conf (float): confidence threshold
        url (str): url to YouTube video
    """
    if url != '':
        preds = model.predict(source=url, stream=True)
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension="mp4", res=720).first()
        vid_cap = cv2.VideoCapture(stream.url)

        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                display_single_frame(
                    conf=conf,
                    model=model,
                    st_frame=st_frame,
                    frame=image,
                )
            else:
                vid_cap.release()
                break
