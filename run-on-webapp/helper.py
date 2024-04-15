import streamlit as st
import os

def save_uploaded_file(uploaded_file):
    with open(os.path.join('../test/', uploaded_file.name + 'saved'), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return st.success("Save file {} to dir".format(uploaded_file.name))

