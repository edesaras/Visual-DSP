import streamlit as st
import numpy as np
import cv2
import io


if __name__ == "__main__":

    source_selectbox = st.selectbox("Select Image Source", ["Take a Photo", "Upload an image"])
    operation_selectbox = st.selectbox("Select Image Transform", ["Flip", "Gaussian Blur", "Grayscale"])

    if source_selectbox == "Take a Photo":
        col1, col2 = st.columns(2)
        with col1:
            image = st.camera_input("Take a Photo")
    elif source_selectbox == "Upload an image":
        image = st.file_uploader("Upload an image", ["png", "jpg"])
        col1, col2 = st.columns(2)
        if image is not None:
            with col1:
                st.image(image, "Original Image")

    # options on sidebar based on the selected operation
    with st.sidebar:
        if operation_selectbox == "Gaussian Blur":
            gaussian_blur_kernel_size = st.sidebar.slider("Gaussian Blur Kernel Size", min_value=3, max_value=21, step=2)

    if image is not None:
        # To read image file buffer with OpenCV:
        bytes_data = image.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # apply operations switch statement
        if operation_selectbox == "Flip":
            img = cv2.flip(img, 1)
        if operation_selectbox == "Grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if operation_selectbox == "Gaussian Blur":
            img = cv2.GaussianBlur(img, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), cv2.BORDER_DEFAULT)

        with col2:
            st.image(img, "Processed Image")

        # encode the image and download button
        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        io_buf = io.BytesIO(buffer)
        download_button = st.download_button(
            label="Download Image",
            data=io_buf.getvalue(),
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )

