import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import pickle

# import model
svm = pickle.load(open('SVC.pkl', 'rb'))

# load heart dataset
data = pd.read_csv('Heart Dataset.csv')

st.title('Aplikasi Heart')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Heart Disease Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1, unsafe_allow_html=True)
activities = ['SVM', 'Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?', activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 = """
    <br>
    <p>Ini adalah dataset Heart Disease</p>
    """
    st.markdown(html_layout2, unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    # Use pandas_profiling or other EDA methods here
    st.subheader('Input Dataframe')
    st.write(data)

# train test split
X = data.drop('age', axis=1)  # Assuming 'target' is the column indicating the output
y = data['age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    # Modify these sliders based on the features in your Heart dataset
    age = st.sidebar.slider('Age', 29, 77, 40)
    sex = st.sidebar.slider('Sex (0 for female, 1 for male)', 0, 1, 1)
    cp = st.sidebar.slider('Chest Pain Type', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 240)
    fbs = st.sidebar.slider('Fasting Blood Sugar (> 120 mg/dl)', 0, 1, 0)
    restecg = st.sidebar.slider('Resting Electrocardiographic Results', 0, 2, 1)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150)
    
    user_report_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': cp,
        'RestingBloodPressure': trestbps,
        'SerumCholesterol': chol,
        'FastingBloodSugar': fbs,
        'RestingECG': restecg,
        'MaxHeartRate': thalach
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Data Pasien
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = svm.predict(user_data)
svc_score = accuracy_score(y_test, svm.predict(X_test))

# Output
st.subheader('Hasilnya adalah : ')
output = ''
if user_result[0] == 0:
    output = 'Kamu Aman'
else:
    output = 'Kamu terkena penyakit jantung'
st.title(output)
st.subheader('Model yang digunakan : \n' + option)
st.subheader('Accuracy : ')
st.write(str(svc_score * 100) + '%')
