import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 


#import model 
svm = pickle.load(open('SVC.pkl','rb'))

#pip install phik#load dataset
data = pd.read_csv('HeartDataset.csv')
#data = data.drop(data.columns[0],axis=1)


st.title('Aplikasi Heart')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Heart Disease Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['KNN','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset PIMA Indian</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('output',axis=1)
y = data['output']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
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
    age = st.sidebar.slider('Age', 0, 77, 40)
    sex = st.sidebar.slider('Sex (0 for female, 1 for male)', 0, 1, 1)
    cp = st.sidebar.slider('Chest Pain Type', 0, 3, 1)
    trstbps = st.sidebar.slider('Resting Blood Pressure', 0, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 0, 564, 240)
    fbs = st.sidebar.slider('Fasting Blood Sugar (> 120 mg/dl)', 0, 1, 0)
    restecg = st.sidebar.slider('Resting Electrocardiographic Results', 0, 2, 1)
    thalachh = st.sidebar.slider('Maximum Heart Rate Achieved', 0, 202, 150)
    thall = st.sidebar.slider ('Thallium Test Result', 0, 3, 1)
    caa = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 3, 0)
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 0.0)
    exng = st.sidebar.slider('Exercise Induced Angina (0 for No, 1 for Yes)', 0, 1, 0)
    slp = st.sidebar.slider('Slope of the Peak Exercise ST Segment', 0, 2, 1)
    
    user_report_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': cp,
        'RestingBloodPressure': trstbps,
        'SerumCholesterol': chol,
        'FastingBloodSugar': fbs,
        'RestingECG': restecg,
        'MaxHeartRate': thalachh,
        'ThalliumTestResult': thall,
        'NumberOfMajorVessels': caa,
        'STDepression': oldpeak,
        'ExerciseInducedAngina': exng,
        'SlopeOfPeakExerciseSTSegment': slp
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = svm.predict(user_data)
svc_score = accuracy_score(y_test,svm.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena penyakit jantung'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')



