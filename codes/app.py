import cv2
import numpy as np
import streamlit as st

def apply_sharpening(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img

def apply_threshold(img, threshold_value):
    _, thresholded_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_img

def apply_contrast_stretching(img, min_out=0, max_out=255):
    min_in, max_in = np.min(img), np.max(img)
    stretched_img = ((img - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out
    return stretched_img.astype(np.uint8)

def main():
    st.title("Image Enhancement with Streamlit")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create sliders for parameter adjustment
        sharpening_flag = st.checkbox("Apply Sharpening", value=True)
        threshold_value = st.slider("Threshold Value", 0, 255, 128)
        min_stretch = st.slider("Min Stretch", 0, 255, 0)
        max_stretch = st.slider("Max Stretch", 0, 255, 255)

        # Apply enhancements based on user inputs
        enhanced_img = img.copy()
        if sharpening_flag:
            enhanced_img = apply_sharpening(enhanced_img)
        enhanced_img = apply_threshold(enhanced_img, threshold_value)
        enhanced_img = apply_contrast_stretching(enhanced_img, min_stretch, max_stretch)

        # Display the original and enhanced images
        st.image([img, enhanced_img], caption=["Original Image", "Enhanced Image"], width=300)
    else:
        st.warning("Please upload an image.")

if __name__ == '__main__':
    main()
