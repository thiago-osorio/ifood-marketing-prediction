import pickle
import pandas as pd
import numpy as np
import datetime
import streamlit as st
from PIL import Image

def predicao(data, modelo):
    try:
        df = pd.read_csv(data, sep=';')
        print(df)
        df['Year_Birth'] = df['Year_Birth'].astype('int')
        df['age'] = 2022 - df['Year_Birth']
        df['today'] = pd.to_datetime(datetime.datetime.today().strftime("%Y-%m-%d"))
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
        df['Enrollment_time'] = (df['today'] - df['Dt_Customer'])/np.timedelta64(1, 'Y')
        df.drop(['Dt_Customer', 'today', 'Year_Birth'], axis=1, inplace=True)
        X = pd.get_dummies(df.drop('ID', axis=1))
        columns = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'age', 'Enrollment_time', 'Education_2n Cycle', 'Education_Basic', 'Education_Graduation', 'Education_Master', 'Education_PhD', 'Marital_Status_Absurd', 'Marital_Status_Alone', 'Marital_Status_Divorced', 'Marital_Status_Married', 'Marital_Status_Single', 'Marital_Status_Together', 'Marital_Status_Widow', 'Marital_Status_YOLO']
        for column in columns:
            if column not in X.columns:
                X[column] = 0
        pred = modelo.predict(X)
        df['Predict'] = pred
        df[['ID', 'Predict']].to_csv('prediction/predict.csv', index=None)
        return 'Predictions made with success'
    except:
        return 'Prediction error'

st.image(Image.open('source/logo.png'), width=120)

st.title('iFood Marketing Prediction')

st.caption('Please, download the csv file and put your customers information')

with open('data/example.csv', 'rb') as file:
    btn = st.download_button(
        label='Download',
        data=file,
        file_name='example.csv'
    )

st.markdown('---')

st.subheader('Now it\'s time to generate prediction')

data = st.file_uploader(
    label = 'Upload csv',
    type='csv'
)

if data is not None:
    modelo = pickle.load(open('models/lightgbm_ifood.pkl', 'rb'))
    pred = predicao(data, modelo)
    st.write(pred)
    if pred == 'Predictions made with success':
        with open('prediction/predict.csv', 'rb') as prediction:
            btn_download = st.download_button(
                label='Results',
                data=prediction,
                file_name='predictions.csv'
            )