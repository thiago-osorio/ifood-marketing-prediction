import pickle
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import scale
import streamlit as st
from PIL import Image

def predicao(data, scaler, modelo):
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
        X_standard = scaler.transform(X)
        pred = modelo.predict(X_standard)
        df['Predict'] = pred
        df['Predict'] = df['Predict'].map(lambda x: 'Will respond' if x == 1 else 'Will NOT respond')
        df[['ID', 'Predict']].to_csv('prediction/predict.csv', index=None, sep=';')
        return 'Predictions made with success'
    except:
        return 'Prediction error'

st.image(Image.open('source/logo.png'), width=120)

st.title('iFood Marketing Prediction')

mode = st.radio(
    'Select what kind of prediction you want',
    ('CSV', 'Screen')
)

if mode == 'CSV':
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
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        pred = predicao(data, scaler, modelo)
        st.write(pred)
        if pred == 'Predictions made with success':
            with open('prediction/predict.csv', 'rb') as prediction:
                btn_download = st.download_button(
                    label='Results',
                    data=prediction,
                    file_name='predictions.csv'
                )
else:
    st.markdown('---')
    
    st.subheader('Now it\'s time to generate prediction')
    
    user_id = st.text_input('User ID')
    year_birth = st.number_input('Year Birth', format='%i', step=1, min_value=1900, value=1900)
    education = st.selectbox('Education', ['Graduation', 'PhD', 'Master Education', 'Basic', '2n Cycle'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Togheter', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'])
    income = st.number_input('Income', format='%i', step=1)
    kidhome = st.number_input('Kidhome', format='%i', step=1, min_value=0, max_value=2)