import streamlit as st
import numpy as np
import pandas as pd
import joblib

df = pd.read_csv('tehranhouses.csv')

ac= df['Address'].unique().tolist()

df = pd.get_dummies(df, columns=['Address'])
columns= df.columns

def preprocessor(df2):
    df2 = pd.get_dummies(df2, columns=['Address'])
    df2 = df2.replace({'True': 1, 'False': 0})
    df2 = df2.reindex(columns=columns, fill_value=0)
    return df2

model = joblib.load('catboost.joblib')
st.title('Prediction of House Price in Tehran')

Area = st.number_input("Enter Area", 20) 
Room = st.selectbox("How many Rooms does it have?", [0,1,2,3,4,5])
Parking = st.selectbox("Does it have parking?", ['True','False'])
Warehouse = st.selectbox("Does it have Warehouse?", ['True','False'])
Elevator = st.selectbox("Does it have Elevator?", ['True','False'])
Address = st.selectbox("Where is it located?", ac) 

def predictprice(Area,Room,Parking,Warehouse,Elevator,Address): 
    row = np.array([Area,Room,Parking,Warehouse,Elevator,Address]) 
    X = pd.DataFrame([row], columns = ['Area','Room','Parking','Warehouse','Elevator','Address'])
    p_x= preprocessor(X)
    prediction = model.predict(p_x)
    st.success(f'{prediction}')

trigger = st.button('Predict', on_click=predictprice, args=(Area, Room, Parking, Warehouse, Elevator, Address))

