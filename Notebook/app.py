import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pycaret.classification import load_model
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.mstats import winsorize

class Cleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Salin data untuk mencegah perubahan pada data asli
        X = X.copy()
        
        # Penerapan Winsorization untuk mengatasi outliers
        X['horsepower'] = winsorize(X['horsepower'])
        X['peakrpm'] = winsorize(X['peakrpm'])
        X['citympg'] = winsorize(X['citympg'])
        X['cylindernumber'] = winsorize(X['cylindernumber'])

        # Kembali sebagai DataFrame agar sesuai dengan nama kolom asli
        return pd.DataFrame(X, columns=X.columns)
    def delate(self,X,y=None):
        X.drop(columns=['CarName','highwaympg','enginesize'],inplace=True)
        return pd.DataFrame(X, columns=X.columns)
    
@st.cache_resource() 
def get_model(): 
    return joblib.load("FinalModel.pkl") 

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    model=get_model()
    st.title("Car Price Prediction App")
    add_selectbox=st.selectbox("Pilih Cara Anda Untuk Melakukan Prediksi?", ("Online" , "Batch"))
    # menambahkan keterangan pada sidebar
    st.sidebar.info('Aplikasi ini digunakan untuk memprediksi harga mobil bekas')
    
    # Menambahkan title
    
    if add_selectbox=="Online":
        Fuel_System=st.selectbox('Fuel System:',('mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi'))
        Fuel_Type=st.selectbox('Jenis Bahan Bakar:',('gas', 'diesel'))
        Engine_Location = st.selectbox('Engine Location:',('front', 'rear'))
        Aspiration = st.selectbox('Aspiration:',('std', 'turbo'))
        Car_Body = st.selectbox('Car Body:',('convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'))
        Drive_wheel = st.selectbox('Drive wheel:',('rwd', 'fwd', '4wd'))
        Engine_Type = st.selectbox('Engine Type:',('dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv'))
        Symboling=st.number_input('Symboling: ',min_value=-2,max_value=3,step=1)
        horsepower=st.number_input('Horsepower:',min_value=48,max_value=288,step=1)
        peakrpm=st.number_input('peak RPM:',min_value=4150,max_value=6600,step=1)
        citympg=st.number_input('cityMpg:',min_value=13,max_value=49,step=1)
        cylindernumber=st.number_input('Cylindr Number:',min_value=2,max_value=12,step=1)

        input_df=pd.DataFrame([
            {
                'fuelsystem': Fuel_System,
                'fueltype': Fuel_Type,
                'enginelocation': Engine_Location,
                'aspiration': Aspiration,
                'carbody': Car_Body,
                'drivewheel': Drive_wheel,
                'enginetype': Engine_Type, 
                'symboling':Symboling,
                'horsepower':horsepower,
                'peakrpm':peakrpm,
                'citympg': citympg,
                'cylindernumber':cylindernumber
            }
        ])

        output = ""

        # # Make a prediction   
        if st.button("Predict"):
            output = model.predict(input_df)[0]  # Pastikan Anda mengambil nilai prediksi tunggal jika model mengembalikan array
            formatted_output = f"${output:,.2f}"  # Format output ke format dolar
            st.success(f"Harga yang diprediksi: {formatted_output}")  # Menampilkan hasil dalam format dolar

    if add_selectbox=="Batch":
        file_Upload=st.file_uploader("Upload CSV file untuk memprediksi", type=["csv"])
        

        if file_Upload is not None:
            data=pd.read_csv(file_Upload)

            #select kolom
            data=data[[
                'fuelsystem',
                'fueltype',
                'enginelocation',
                'aspiration',
                'carbody',
                'drivewheel',
                'enginetype', 
                'symboling',
                'horsepower',
                'peakrpm',
                'citympg',
                'cylindernumber'
            ]]

            #Prediction
            prediksi= model.predict(data)
            data['Prediction'] = [f"${pred:,.2f}" for pred in prediksi]

            # menampilkan hasi prediksi
            st.write(data)

            #Menambahkan button download 
            st.download_button(
                "Press button untuk mendownload",convert_df(data),"Hasil prediksi.csv","text/csv",key='download-csv')
    
if __name__ == '__main__':
    main()

