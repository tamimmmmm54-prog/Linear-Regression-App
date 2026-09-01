import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

