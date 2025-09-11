import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Judul aplikasi
st.title("Prediksi Skor Kredit dengan Logistic Regression")

# Memuat model dan scaler
try:
    lr_model = joblib.load('lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("Model dan scaler berhasil dimuat.")
except FileNotFoundError:
    st.error("Error: File 'lr_model.pkl' atau 'scaler.pkl' tidak ditemukan.")
    st.stop()

# Memuat data pelatihan untuk referensi kolom
try:
    df_train = pd.read_csv('balanced_data.csv')
except FileNotFoundError:
    st.error("Error: File 'balanced_data.csv' tidak ditemukan.")
    st.stop()

# Kolom numerik dan kategorikal
numeric_cols = ['PRICE_CAR', 'DOWN_PAYMENT', 'PERCENT_DOWN_PAYMENT', 'INTEREST_RATE', 
                'AMT_CREDIT', 'AMT_ANNUITY', 'INCOME_TOTAL', 'TENOR', 'RATIO_DTI', 
                'AGE', 'TOTAL_DEPENDENTS', 'CREDIT_TO_INCOME']
categorical_cols = ['DATA_BLACKLIST', 'DEBITUR', 'NAME_CAR', 'OCCUPATION_TYPE', 
                    'NAME_HOUSING_TYPE', 'PARTNER_OCCUPATION']
one_hot_cols = ['NAME_CAR', 'OCCUPATION_TYPE', 'NAME_HOUSING_TYPE', 'PARTNER_OCCUPATION']
train_columns = [col for col in df_train.columns if col not in ['SCOR', 'SCOR_BINARY']]

# Mapping untuk label encoding
label_mappings = {
    'DATA_BLACKLIST': {'Ada': 1, 'Tidak': 0},
    'DEBITUR': {'Baru': 1, 'Lama': 0}
}

# Form input
st.header("Masukkan Data Baru")
with st.form(key='prediction_form'):
    # Input numerik
    price_car = st.number_input("Harga Mobil (PRICE_CAR)", min_value=0.0, value=200000000.0)
    down_payment = st.number_input("Uang Muka (DOWN_PAYMENT)", min_value=0.0, value=50000000.0)
    percent_down_payment = st.number_input("Persentase Uang Muka (%)", min_value=0.0, value=25.0)
    interest_rate = st.number_input("Suku Bunga (%)", min_value=0.0, value=5.0)
    amt_credit = st.number_input("Jumlah Kredit (AMT_CREDIT)", min_value=0.0, value=150000000.0)
    amt_annuity = st.number_input("Angsuran per Tahun (AMT_ANNUITY)", min_value=0.0, value=5000000.0)
    income_total = st.number_input("Total Pendapatan (INCOME_TOTAL)", min_value=0.0, value=10000000.0)
    tenor = st.number_input("Tenor (bulan)", min_value=0, value=36, step=1)
    ratio_dti = st.number_input("Rasio DTI (%)", min_value=0.0, value=50.0)
    age = st.number_input("Usia (AGE)", min_value=0, value=30, step=1)
    total_dependents = st.number_input("Jumlah Tanggungan (TOTAL_DEPENDENTS)", min_value=0, value=1, step=1)

    # Input kategorikal
    data_blacklist = st.selectbox("Data Blacklist", options=['Ada', 'Tidak'])
    debitur = st.selectbox("Debitur", options=['Baru', 'Lama'])
    name_car = st.selectbox("Tipe Mobil (NAME_CAR)", options=df_train['NAME_CAR'].unique())
    occupation_type = st.selectbox("Tipe Pekerjaan (OCCUPATION_TYPE)", options=df_train['OCCUPATION_TYPE'].unique())
    name_housing_type = st.selectbox("Tipe Perumahan (NAME_HOUSING_TYPE)", options=df_train['NAME_HOUSING_TYPE'].unique())
    partner_occupation = st.selectbox("Pekerjaan Pasangan (PARTNER_OCCUPATION)", options=df_train['PARTNER_OCCUPATION'].unique())

    submit_button = st.form_submit_button(label='Prediksi')

# Proses prediksi saat form disubmit
if submit_button:
    try:
        # Buat DataFrame dari input
        new_data = {
            'PRICE_CAR': price_car,
            'DOWN_PAYMENT': down_payment,
            'PERCENT_DOWN_PAYMENT': percent_down_payment,
            'INTEREST_RATE': interest_rate,
            'AMT_CREDIT': amt_credit,
            'AMT_ANNUITY': amt_annuity,
            'INCOME_TOTAL': income_total,
            'TENOR': tenor,
            'RATIO_DTI': ratio_dti,
            'AGE': age,
            'TOTAL_DEPENDENTS': total_dependents,
            'DATA_BLACKLIST': data_blacklist,
            'DEBITUR': debitur,
            'NAME_CAR': name_car,
            'OCCUPATION_TYPE': occupation_type,
            'NAME_HOUSING_TYPE': name_housing_type,
            'PARTNER_OCCUPATION': partner_occupation
        }
        new_df = pd.DataFrame([new_data])

        # Tambahkan fitur CREDIT_TO_INCOME
        new_df['CREDIT_TO_INCOME'] = new_df['AMT_CREDIT'] / new_df['INCOME_TOTAL']

        # Label encoding
        new_df['DATA_BLACKLIST'] = new_df['DATA_BLACKLIST'].map(label_mappings['DATA_BLACKLIST'])
        new_df['DEBITUR'] = new_df['DEBITUR'].map(label_mappings['DEBITUR'])

        # One-hot encoding
        new_df_encoded = pd.get_dummies(new_df, columns=one_hot_cols)

        # Pastikan kolom sesuai dengan data pelatihan
        for col in train_columns:
            if col not in new_df_encoded.columns:
                new_df_encoded[col] = 0
        new_df_encoded = new_df_encoded[train_columns]

        # Standarisasi numerik
        new_df_encoded[numeric_cols] = scaler.transform(new_df_encoded[numeric_cols])

        # Prediksi
        pred_lr = lr_model.predict(new_df_encoded)[0]
        pred_proba_lr = lr_model.predict_proba(new_df_encoded)[0]
        prediction = 'Baik' if pred_lr == 1 else 'Buruk'
        proba_buruk = round(pred_proba_lr[0], 4)
        proba_baik = round(pred_proba_lr[1], 4)

        # Tampilkan hasil
        st.header("Hasil Prediksi")
        st.write(f"**Prediksi:** {prediction}")
        st.write(f"**Probabilitas Buruk:** {proba_buruk}")
        st.write(f"**Probabilitas Baik:** {proba_baik}")
        st.subheader("Data Input:")
        for key, value in new_data.items():
            st.write(f"{key}: {value}")

    except Exception as e:
        st.error(f"Error: {str(e)}. Pastikan input valid dan sesuai dengan format.")