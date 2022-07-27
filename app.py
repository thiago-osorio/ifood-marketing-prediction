import pickle
import pandas as pd
import streamlit as st
from PIL import Image

modelo = pickle.load(open('models/lightgbm_ifood.pkl', 'rb'))

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