import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle


## load trained model
model = tf.keras.models.load_model('model.h5')

with open('one_encoder.pkl','rb') as file:
    one_encoder = pickle.load(file)

with open('label.pkl','rb') as file:
    label = pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar = pickle.load(file)


## streamlit app

st.title('Customer Churn Prediction ')

## user input

geography = st.selectbox('Geography',one_encoder.categories_[0])
gender = st.selectbox('Gender',label.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Prodcuts',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1]) 

## prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]

})
## geography one hot encoding
geo_encoder = one_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoder,columns=one_encoder.get_feature_names_out(['Geography']))

# Combine one hot encoded value with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# scaled data
input_data_scaled = scalar.transform(input_data)

## predict churn

prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn probablity:{prediction_prob* 100:.2f}%')

if prediction_prob>0.5:
    st.write('Customer likely to be churn')
else:
    st.write('Customer not likely to be churn')
