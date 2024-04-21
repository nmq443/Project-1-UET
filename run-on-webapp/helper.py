import streamlit as st
import os
import cv2
from PIL import Image
from pytube import YouTube

def save_uploaded_file(uploaded_file):
    with open(os.path.join('../test/', uploaded_file.name + 'saved'), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return st.success("Save file {} to dir".format(uploaded_file.name))

def image_object_detection(image, conf, model):
    results = model.predict(image, conf=conf)

    for result in results:
        res_bgr = result.plot()
        res_rgb = Image.fromarray(res_bgr[..., ::-1] )
        st.image(
            image=res_rgb,
            caption='Predicted image'
        )

def display_single_frame(model, conf, st_frame, frame): 
    results = model.track(frame, conf=conf, persist=True)
    frame_bgr = results[0].plot()
    frame_rgb = Image.fromarray(frame_bgr[..., ::-1])

    # visualization
    st_frame.image(frame_rgb)

def realtime_object_detection(video_src, conf, model):
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
