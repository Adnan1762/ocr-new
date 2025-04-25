App Link:
https://textvisionnew.streamlit.app/
# TextVision OCR

TextVision OCR is a powerful Optical Character Recognition (OCR) web application designed to accurately extract text from images in various formats such as PNG, JPG, and JPEG. Utilizing PaddleOCR, the application effectively recognizes text across different languages and presents it in a user-friendly interface. The project not only extracts text but also provides a visual representation of detected text through bounding boxes, allowing users to verify the accuracy of the recognition. Additionally, users can easily download the extracted text as a .txt file for further use.

## Features:

Multi-Language Support: Leverages PaddleOCR's capabilities to recognize text in multiple languages. Users can configure the language parameter to accommodate diverse linguistic needs.

User-Friendly Interface: Built with Streamlit, the application offers a clean and intuitive interface, making it easy for users to upload images and view results without technical hurdles.

Text Extraction with Visual Feedback:As the application processes the uploaded image, it displays the detected text along with bounding boxes around the recognized areas, enhancing user understanding and verification of results.

Confidence Scores: Each extracted text element is accompanied by a confidence score, providing insight into the accuracy of the OCR results and helping users assess the reliability of the extraction.

Downloadable Output: Users can download the extracted text in a .txt format, facilitating easy access and further manipulation of the data for their projects or needs.

How to run this code:
Step 1- pip install -r requirements.txt
Step 2- streamlit run app.py

TextVision OCR combines advanced OCR technology with user-centric design to deliver a robust solution for text extraction tasks. Whether for academic purposes, business applications, or personal projects, this tool provides essential features to meet various user needs in a seamless manner.
