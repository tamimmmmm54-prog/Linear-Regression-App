import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

st.title('Linear Regression Web Aplication')
st.subheader('Data Science with Sifat Ahmed Tamim')

# Sidebar
st.sidebar.header('Upload CSV Data or Use Sample')
use_example= st.sidebar.checkbox('Use example Dataset')

#Load Dataset
if use_example:
  df = sns.load_dataset('tips')
  df = df.dropna()
  st.success("Loaded simple Dataset:'tips'")
else:
  uploaded_file = st.sidebar.file_uploader('Upload your CSV_file',type=['csv'])
  if uploaded_file:
    df = pd.read_csv(uploaded_file)
  else:
    st.warning('Please upload a csv file or use the example dataset')
    st.stop()

#Show dataset
st.subheader('Dataset Preview')
st.write(df.head())
    
# feature selection and model training
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
  st.error('Please at least two numeric columns for regression')
  st.stop()

target = st.selectbox('Select Target varible',numeric_cols)
features = st.multiselect('Select Input Feature Columns',[col for col in numeric_cols if col !=target],default=[col for col in numeric_cols if col !=target])
if len(features) == 0:
  st.write('Please Select at least one feature')
  st.stop()

df = df[features + [target]].dropna()
x = df[features]
y = df[target]

scaler = StandardScaler()
X_scaler = scaler.fit_transform(x)
xtrain,xtest,ytrain,ytest = train_test_split(X_scaler,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)
mse = mean_squared_error(ytest,y_pred)
r2 = r2_score(ytest,y_pred)
st.subheader('Model evaluation')
st.write(f'Mean Squared Error:{mse:.2f}')
st.write(f'R*2 Score:{r2:.2f}')

st.subheader('Make a Prediction')
input_data = {}
valid_input = True
for feature in features:
  user_val = st.text_input(f'Enter {feature} (numeric value)')
  try:
    if user_val.strip()=='':
      valid_input = False
    else:
      input_data[feature] = float(user_val)
  except ValueError:
    valid_input = False

if st.button('Predict'):
  if valid_input:
    input_df = pd.DataFrame([input_data]):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f'Predicted {target}: {Prediction[0]:.2f}')
  else 

st.error('Please Enter Valid numeric values for all features before prediction')


