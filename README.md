
# Lung Disease Prediction System

## Working Link
- http://34.93.183.143/


## Demo
https://github.com/AdityaCoder-N/LungsDiseasePredictionSystem/assets/91337266/843c28d3-4546-469a-b880-23dd40bb29f9


## Overview
The Lung Disease Prediction System is a web application that uses deep learning models to predict lung diseases, specifically Tuberculosis and Pneumonia, from X-ray images of patients. The system is built using Streamlit for the frontend and TensorFlow/Keras for the deep learning models.

## Features
- Tuberculosis Prediction: Upload an X-ray image of a patient's lung to predict whether they have Tuberculosis or not.
- Pneumonia Prediction: Upload an X-ray image of a patient's lung to predict whether they have Pneumonia or not.

## Installation
1. Clone the repository:
git clone https://github.com/your-username/lung-disease-prediction.git
cd lung-disease-prediction

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate


3. Install the required packages:
pip install -r requirements.txt


## Usage
1. Run the Streamlit app locally:
streamlit run interface.py


2. Access the app in your web browser by opening the provided link (usually `http://localhost:8501`).

3. Select the Lung Disease Prediction System you want to use (Tuberculosis or Pneumonia).

4. Upload an X-ray image of a patient's lung.

5. Click the "Predict" button to get the prediction and confidence level for the selected lung disease.

## Model Details
- The Tuberculosis prediction model has an input shape of (32, 256, 256, 3) and uses a Convolutional Neural Network (CNN) architecture with multiple convolutional and max-pooling layers.
- The Pneumonia prediction model has an input shape of (256, 256, 1) and uses a CNN architecture with batch normalization and dropout layers.

## Dataset
- The X-ray images used for training and evaluating the models were obtained from the public dataset 
- Tuberculosis dataset : https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities
- Pneumonia dataset : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia


## Contact
- Name:Aditya Negi negiaditya1234@gmail.com

