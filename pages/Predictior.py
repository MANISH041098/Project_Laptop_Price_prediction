import streamlit as st
import numpy as np
from pickle import load

df = load(open('df2.pkl','rb'))
lr_model = load(open('laptop_analysis.pkl','rb'))
OHE = load(open('OHE.pkl','rb'))

st.header('Laptop Predictor')

Brand = st.selectbox('Brand',df['Brand'].unique())
Processor = st.selectbox('Processor',df['Processor'].unique())
RAM = st.selectbox('RAM',df['RAM'].unique())
RAM_type = st.selectbox('RAM type',df['RAM type'].unique())
OS = st.selectbox('OS',df['OS'].unique())
Storage = st.selectbox('Storage',df['Storage'].unique())
Screensize = st.selectbox('Screensize',df['Screensize'].unique())

btn_click = st.button('Predict')

if btn_click == True:
    query = np.array([Processor,RAM,OS,Storage,Brand,Screensize,RAM_type]).reshape(1,-1)
    query_transformed = OHE.transform(query)
    pred = lr_model.predict(query_transformed)
    st.success("The predicted price is "+ str(int(np.exp(pred))))
else:
    st.error('not found')
