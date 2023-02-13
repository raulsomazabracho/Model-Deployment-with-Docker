import pickle
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import numpy as np

# Load the normalization constants from the file
with open('model_normalization_constants.pkl', 'rb') as f:
    constants = pickle.load(f)

age_scaler_min = constants['age_scaler_min']
age_scaler_max = constants['age_scaler_max']
bmi_scaler_min = constants['bmi_scaler_min']
bmi_scaler_max = constants['bmi_scaler_max']
ch_scaler_min = constants['ch_scaler_min']
ch_scaler_max = constants['ch_scaler_max']

# Page Title
st.title("Medical ensurande calculator (Machine Learning Based)")
st.subheader("We will calculate your ensurance cost based of a Nerural Network")

# User Features
name = st.text_input("Please insert your name")
age = st.slider("How old are you?", 1, 100, 1)
bmi = st.slider("Insert Your BMI", 1, 100, 1)
ch = st.slider("How many children de you have?", 0, 10, 1)
sex = st.selectbox("Insert Your Genre", options=['Male' , 'Female'])
smoker= st.selectbox("Are you a smoker?", options=['Yes' , 'No'])
region = st.selectbox("Insert Your Region", options=['Northeast' , 'Northwest', 'Southeast', 'Southwest'])

#Data Manipulation
age_scaled = (age - age_scaler_min ) / (age_scaler_max - age_scaler_min )
bmi_scaled = (bmi - bmi_scaler_min ) / (bmi_scaler_max - bmi_scaler_min )
ch_scaled = (ch - ch_scaler_min ) / (ch_scaler_max - ch_scaler_min )
input_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
input_data[0][0]  = age_scaled[0]
input_data[0][1]  = bmi_scaled[0]
input_data[0][2]  = ch_scaled[0]
input_data[0][3]  = 1 if sex == 'Female' else 0
input_data[0][4]  = 1 if sex == 'Male' else 0
input_data[0][5]  = 1 if smoker == 'No' else 0
input_data[0][6]  = 1 if smoker == 'Yes' else 0
input_data[0][7]  = 1 if region == 'Northeast' else 0
input_data[0][8]  = 1 if region == 'Northwest' else 0
input_data[0][9]  = 1 if region == 'Southeast' else 0
input_data[0][10]  = 1 if region == 'Southwest' else 0

# load the model from disk
loaded_model = pickle.load(open('NN_model.sav', 'rb'))

#Prediction
prediction = loaded_model.predict(input_data)

#Results
st.subheader('{}, your ensurance estimated cost is {}$'.format(name, round(prediction[0] , 3)))

