import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from yolo import process_yolo
from red import red
from white import white

def segment_image(image, mode):
    if mode == "Red eggs":
        return red(image)
    elif mode == "White eggs":
        return white(image)

def display_carousel(images, captions, key_prefix):
    if f"current_index_{key_prefix}" not in st.session_state:
        st.session_state[f"current_index_{key_prefix}"] = 0

    st.subheader("Image segmentation process")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Prev", key=f"prev_button_{key_prefix}"):
            st.session_state[f"current_index_{key_prefix}"] = max(0, st.session_state[f"current_index_{key_prefix}"] - 1)
    with col2:
        if st.button("Next", key=f"next_button_{key_prefix}"):
            st.session_state[f"current_index_{key_prefix}"] = min(len(images) - 1, st.session_state[f"current_index_{key_prefix}"] + 1)

    st.image(images[st.session_state[f"current_index_{key_prefix}"]], 
             caption=captions[st.session_state[f"current_index_{key_prefix}"]], 
             width=500)

def clear_session_state():
    keys_to_keep = ['carosel', 'captions', 'current_index']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

def main():
    st.title("Egg segmentation")
    st.sidebar.title("Menu")
    if 'prev_model' not in st.session_state:
        st.session_state.prev_model = None
    model_type = st.sidebar.radio("Choose model", ["YOLO", "U-NET"])
    if model_type != st.session_state.prev_model:
        clear_session_state()
        st.session_state.prev_model = model_type

    if model_type == "U-NET":
        mode = st.sidebar.radio("Choose mode", ["Red eggs", "White eggs"])
    else:
        mode = None

    uploaded_file = st.sidebar.file_uploader("Upload your image", 
                                           type=["jpg", "jpeg", "png"])

    folder_path = "data"
    if os.path.exists(folder_path):
        sample_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        sample_file = st.sidebar.selectbox("Or select sample image", 
                                         [None] + sample_files)
    else:
        sample_files = []
        sample_file = None

    if uploaded_file is not None or sample_file is not None:
        if st.sidebar.button("Segment"):
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = np.array(image)
            else:
                image_path = os.path.join(folder_path, sample_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (256, 256))

            if model_type == "U-NET":
                carosel, captions = segment_image(image, mode)
            else:
                carosel, captions = process_yolo(image)
            
            st.session_state.carosel = carosel
            st.session_state.captions = captions
            st.session_state.current_index = 0

        if "carosel" in st.session_state and "captions" in st.session_state:
            display_carousel(st.session_state.carosel, st.session_state.captions, key_prefix="single")

if __name__ == "__main__":
    main()