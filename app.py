import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle

# Judul aplikasi
st.title("Prediksi Risiko Kredit Mobil Mazda")

# Deskripsi
st.write("""
### Aplikasi Prediksi Risiko Kredit
Masukkan data debitur untuk memprediksi risiko kredit (0 = Lancar, 1 = Macet) menggunakan Logistic Regression dan XGBoost.
""")

# Input data dari user
st.sidebar.header("Input Data Debitur")
price_car = st.sidebar.number_input("Harga Mobil (Rp, jutaan)", min_value=200.0, max_value=600.0, value=371.0)
ratio_dti = st.sidebar.number_input("Rasio DTI (%)", min_value=0.0, max_value=100.0, value=25.0)
income_total = st.sidebar.number_input("Total Pendapatan (Rp, jutaan)", min_value=5.0, max_value=50.0, value=15.0)
total_dependents = st.sidebar.number_input("Jumlah Tanggungan", min_value=0, max_value=10, value=2)
tenor = st.sidebar.number_input("Tenor (tahun)", min_value=1, max_value=5, value=3)

# Tombol prediksi
if st.sidebar.button("Prediksi"):
    # Buat dataframe dari input
    input_data = pd.DataFrame({
        'PRICE_CAR': [price_car],
        'RATIO_DTI': [ratio_dti],
        'INCOME_TOTAL': [income_total],
        'TOTAL_DEPENDENTS': [total_dependents],
        'TENOR': [tenor]
    })

    # Load model (asumsi sudah dilatih dan disimpan)
    with open('lr_model.pkl', 'rb') as file:
        log_reg = pickle.load(file)
    with open('xgb_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)

    # Load scaler (asumsi sudah difit pada train set)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Standarisasi input data
    input_data_scaled = scaler.transform(input_data)

    # Prediksi
    pred_log = log_reg.predict(input_data_scaled)[0]
    pred_prob_log = log_reg.predict_proba(input_data_scaled)[0][1]  # Probabilitas kelas 1
    pred_xgb = xgb_model.predict(input_data_scaled)[0]
    pred_prob_xgb = xgb_model.predict_proba(input_data_scaled)[0][1]  # Probabilitas kelas 1

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    st.write(f"**Logistic Regression:** Prediksi = {'Macet' if pred_log == 1 else 'Lancar'}, Probabilitas Macet = {pred_prob_log:.2f}")
    st.write(f"**XGBoost:** Prediksi = {'Macet' if pred_xgb == 1 else 'Lancar'}, Probabilitas Macet = {pred_prob_xgb:.2f}")

    # Visualisasi sederhana (bar chart probabilitas)
    st.subheader("Visualisasi Probabilitas")
    prob_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost'],
        'Probabilitas Macet': [pred_prob_log, pred_prob_xgb]
    })
    st.bar_chart(prob_data.set_index('Model'))

# Catatan
st.write("""
### Catatan
- Model dilatih pada dataset kredit mobil Mazda dengan 5 fitur utama.
- Pastikan input sesuai dengan rentang data asli untuk hasil optimal.
""")

