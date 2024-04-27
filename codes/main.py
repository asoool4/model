from flask import Flask, request, send_file
import cv2
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Function to apply Image Sharpening
def apply_sharpening(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

# Function to apply Thresholding
def apply_threshold(img, threshold_value):
    _, thresholded_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_img

# Function to apply Contrast Stretching
def apply_contrast_stretching(img, min_out=70, max_out=255):
    min_in, max_in = np.min(img), np.max(img)
    stretched_img = ((img - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out
    return stretched_img.astype(np.uint8)

# Function to process the image
def process_image(img):
    # Apply Sharpening, Thresholding, and Contrast Stretching
    sharpened_img = apply_sharpening(img)
    thresholded_img = apply_threshold(sharpened_img, 120) # Example threshold value
    contrast_stretched_img = apply_contrast_stretching(thresholded_img)
    return contrast_stretched_img

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'file' not in request.files:
        print('No file part')
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return 'No selected file', 400
    if file:
        # Read the image
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # Process the image
        enhanced_img = process_image(img)
        # Convert the enhanced image to a byte stream
        is_success, im_buf_arr = cv2.imencode(".jpg", enhanced_img)
        byte_im = io.BytesIO(im_buf_arr.tobytes())
        return send_file(byte_im, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
