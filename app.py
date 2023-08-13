import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('catboost.joblib')
st.title('Prediction of House Price in Tehran')

Area = st.number_input("Enter Area", '100') 
Room = st.selectbox("How many Rooms does it have?", [0,1,2,3,4,5])
Parking = st.selectbox("Does it have parking?", ['True','False'])
Warehouse = st.selectbox("Does it have Warehouse?", ['True','False'])
Elevator = st.selectbox("Does it have Elevator?", ['True','False'])
Address = st.text_input("Where is it located?", "Punak") 

def predict(): 
    row = np.array([Area,Room,Parking,Warehouse,Elevator,Address]) 
    X = pd.DataFrame([row], columns = [Area,Room,Parking,Warehouse,Elevator,Address])
    prediction = model.predict(X)
    return prediction

trigger = st.button('Predict', on_click=predict)

