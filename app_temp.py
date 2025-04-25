import numpy as np
import pandas as pd
import easyocr
import streamlit as st
from PIL import Image
import cv2
import base64
import pyttsx3  # Text-to-Speech library
import io

# Set the page config with a custom favicon
st.set_page_config(page_title="OCR App", page_icon="generative-image.ico")

# Function to add app background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(f"""<style>.stApp {{background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover}}</style>""", unsafe_allow_html=True)

# Function to draw bounding boxes on image
def display_ocr_image(img, results):
    img_np = np.array(img)

    for detection in results:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        font = cv2.FONT_HERSHEY_COMPLEX

        cv2.rectangle(img_np, top_left, bottom_right, (0, 255, 0), 5)
        cv2.putText(img_np, text, top_left, font, 1, (125, 29, 241), 2, cv2.LINE_AA)

    # Return processed image
    return img_np

# Function to combine all predicted text
def extracted_text(col):
    return " , ".join(img_df[col])

# Text-to-Speech (TTS)
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Download text
def download_text(text):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="extracted_text.txt">Download Extracted Text</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to download image
def download_image(img, img_name):
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    b64_img = base64.b64encode(img_byte_arr.read()).decode()

    href = f'<a href="data:image/png;base64,{b64_img}" download="{img_name}.png">Download Image</a>'
    st.markdown(href, unsafe_allow_html=True)

# Extract embedded images from uploaded image (using contour detection)
def extract_images_from_uploaded_image(image, target_size=(640, 480)):
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply threshold to get a binary image
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_images = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small areas (this threshold can be adjusted)
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the region from the original image
            cropped_img = image.crop((x, y, x + w, y + h))

            # Resize cropped image to the target size (480p)
            cropped_img_resized = cropped_img.resize(target_size)

            # Append resized image to the list
            extracted_images.append(cropped_img_resized)

    return extracted_images

# Streamlit app
st.markdown("""<svg width="600" height="100">
        <text x="50%" y="50%" font-family="monospace" font-size="42px" fill="Turquoise" text-anchor="middle" stroke="white"
         stroke-width="0.3" stroke-linejoin="round"> 🌐 TextVision OCR 🔍📃
        </text>
    </svg>
""", unsafe_allow_html=True)

add_bg_from_local('pic3.jpg')

# Supported languages
languages_supported = ['en', 'hi', 'es', 'fr', 'de', 'zh', 'ja', 'ar', 'it', 'pt']
selected_languages = st.multiselect("Select the language(s) for OCR", options=languages_supported, default=['en'])

# Upload section
file = st.file_uploader(label="Upload Image Here (png/jpg/jpeg):", type=['png', 'jpg', 'jpeg'])

if file is not None:
    # Open the uploaded image
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated to use_container_width

    # Initialize EasyOCR reader with selected languages
    reader = easyocr.Reader(selected_languages, gpu=False)
    results = reader.readtext(np.array(image))

    img_df = pd.DataFrame(results, columns=['bbox', 'Predicted Text', 'Prediction Confidence'])

    # Combine all extracted text
    text_combined = extracted_text(col='Predicted Text')
    st.write("Text Generated :- ", text_combined)

    # Show OCR table
    st.write("Table Showing Predicted Text and Prediction Confidence:")
    st.table(img_df.iloc[:, 1:])

    # Get the final processed image with bounding boxes
    processed_image = display_ocr_image(image, results)
    
    # Display the processed image with bounding boxes
    st.image(processed_image, channels="BGR", use_container_width=True)  # Updated to use_container_width

    # Text-to-Speech Feature
    if st.button("Play Extracted Text"):
        text_to_speech(text_combined)

    # Download extracted text
    download_text(text_combined)

    # Download original image
    download_image(image, "original_image")

    # Download processed image
    download_image(Image.fromarray(processed_image), "processed_image")

    # Extract and allow downloading of images embedded inside the uploaded image
    extracted_images = extract_images_from_uploaded_image(image)
    if extracted_images:
        st.subheader("Extracted Embedded Images:")
        for idx, img in enumerate(extracted_images):
            st.image(img, caption=f"Extracted Image {idx + 1}", use_container_width=True)  # Updated to use_container_width
            download_image(img, f"embedded_image_{idx + 1}")
    else:
        st.write("No embedded images found.")

else:
    st.warning("!! Please Upload your image !!")
