import os #py libaray that provides functions to interact with the OS for (R/W)

import streamlit as st #Streamlit Allows us to create web apps for machine learning. 

import numpy as np #Library for py that adds Multi-dimensional arrays and matrices.

import tensorflow as tf #Library library that is focused on develop and train machine learning models

from keras.preprocessing import image #Allows us to preprocess images data and feeding it machine learning model.

from PIL import Image, ImageDraw, ImageFont #Allows us to add image processing capabilities to our python interpreter.

import cv2 #Computer vision Libraray that is used in wide range of applications for image and video processing and object detection

import zipfile #Library allows us to create, read, write and extract data from zip files.


loaded_model = tf.keras.models.load_model('C:\\Users\\gft10\\Models\\emotions.h5') #Enter the directory of the h5 included in the project files

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Neutral", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Fearful"}


# Define a list of colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Create a file uploader ('More types can be added')
uploaded_files = st.file_uploader("Choose up to 5 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Create a ZIP file to store the processed images
with zipfile.ZipFile('processed_images.zip', 'w') as zipf:
    # Process the uploaded files
    for i, uploaded_file in enumerate(uploaded_files[:5]):
        # Load the image from the uploaded file
        img = Image.open(uploaded_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Load the face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces in the image
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Create an ImageDraw object to draw on the image
        draw = ImageDraw.Draw(img)
        
        # Process each detected face 
        for j, (x, y, w, h) in enumerate(faces):
            # Choose a color for this face
            color = colors[j % len(colors)]
            
            # Draw a rectangle around the face 
            draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=2)
            
            # Extract the face region from the grayscale image -> It's grayscale because colormap is 1
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_array = image.img_to_array(roi_gray)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make a prediction using the loaded model
            prediction = loaded_model.predict(img_array)
            maxindex = int(np.argmax(prediction))
            
            # Display the prediction on the image
            font = ImageFont.truetype("arial.ttf", size=30)
            draw.text((x+20, y-60), emotion_dict[maxindex], fill=color, font=font)
         
        
        # Save the processed image to disk and add it to the ZIP file
        img.save(f'processed_image_{i}.png')
        zipf.write(f'processed_image_{i}.png')
        
        # Display the processed image
        st.image(img)

# Allows the user to download the ZIP file with all processed images
with open('processed_images.zip', 'rb') as f:
    st.download_button('Download images', f.read(), 'images.zip', 'application/zip')
