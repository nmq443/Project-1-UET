import streamlit as st
import os
from PIL import Image

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

