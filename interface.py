import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

image_size = 256
batch_size=32

# Predict pneumonia from the image
def predict_pneumonia(image):
   
    resized_image = image.resize((256, 256))
    
 
    grayscale_image = resized_image.convert("L")
    

    image_array = np.array(grayscale_image)
    

    normalized_image = image_array / 255.0
    
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)
    reshaped_image = np.reshape(preprocessed_image, (-1, 256, 256, 1))
   
    prediction = pn_model.predict(reshaped_image)
    print("ye raha bhai predictions: ",prediction)
    # Get the predicted class

    predicted_class=""
    confidence=0

    if(prediction[0][0] <=0.5):
        predicted_class = "Pneumonia Infected"
        confidence = 100.00-round(100 *(np.max(prediction[0]) ),2)
    else:
        predicted_class = "Healthy Lung"
        confidence = round(100 *(np.max(prediction[0]) ),2)

    return predicted_class , confidence

def predict_tb(img):

    img = img.resize((image_size, image_size))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = tf.keras.preprocessing.image.img_to_array(np.array(img))
    img_array = tf.expand_dims(img_array,0) #create a batch
    
    predictions = tb_model.predict(img_array)
  
    print("predictions : " , predictions)
    predicted_class=""
    confidence=0

    if(predictions[0][0] <=0.5):
        predicted_class = "Healthy Lung" 
        confidence = 100.0 - round(100 *(np.max(predictions[0]) ),2)
    else:
        predicted_class = "Tuberculosis Infected" 
        confidence = round(100 *(np.max(predictions[0]) ),2)
    
    return predicted_class , confidence


# Load the trained models
tb_model = tf.keras.models.load_model('./saved_models/tuberculosis_model.h5')
pn_model = tf.keras.models.load_model('./saved_models/pneumonia_model.h5')


# Sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        'Lung Disease Prediction System',
        ['Tuberculosis Prediction', 'Pneumonia Prediction']
    )

# Function to predict based on the selected disease
def predict(selected, image):
    if selected == 'Tuberculosis Prediction':
        return predict_tb(image)
    elif selected == 'Pneumonia Prediction':
        return predict_pneumonia(image)

# Function to display default images based on the selected disease
def display_default_images(selected, folder_path):
    default_images = os.listdir(folder_path)
    selected_image = st.selectbox(f"Select Default {selected} Image:", default_images, index=0)
    image_path = os.path.join(folder_path, selected_image)
    image = Image.open(image_path)
    st.image(image, caption=f'Default {selected} X-ray Image', use_column_width=True)
    return image

# Tuberculosis or Pneumonia Prediction Page
if selected in ['Tuberculosis Prediction', 'Pneumonia Prediction']:
    st.title(f'{selected} using CNN')

    # File uploader for X-ray image
    uploaded_file = st.file_uploader(f"Upload {selected} X-ray image", type=["png", "jpg", "jpeg"])

    # Use different default image folders based on the selected disease
    default_image_folder = 'default_images_tb' if selected == 'Tuberculosis Prediction' else 'default_images_pn'

    # Add option to choose default images
    if st.checkbox(f"Use Default {selected} Image"):
        image = display_default_images(selected, default_image_folder)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f'Uploaded {selected} X-ray Image', use_column_width=True)

    # Button to perform prediction
    if st.button('Predict'):
        predicted_class, confidence = predict(selected, image)
        st.markdown("<h4 style='text-align: center; color: black;'>Prediction: <strong>{}</strong></h4>".format(predicted_class),
                    unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: black;'>Confidence: <strong>{}%</strong></h4>".format(confidence),
                    unsafe_allow_html=True)