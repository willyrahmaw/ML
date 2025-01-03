import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model dan scaler
@st.cache_data
def load_model(file_path):
    return tf.keras.models.load_model(file_path)  # Untuk model neural network

@st.cache_data
def load_scaler_and_encoder(scaler_path):
    scaler = joblib.load(scaler_path)  # Untuk scaler
    return scaler

# Fungsi untuk prediksi ikan
def predict_fish(model, scaler, input_data):
    # Scaling data input sebelum prediksi
    input_data_scaled = scaler.transform([input_data])  # Pastikan input sudah sesuai dengan scaler

    # Prediksi kelas ikan
    prediction = model.predict(input_data_scaled)
    
    # Ambil indeks kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Tentukan nama kelas berdasarkan indeks
    class_names = [
        'Anabas testudineus', 'Coilia dussumieri', 'Otolithoides biauritus',
        'Otolithoides pama', 'Pethia conchonius', 'Polynemus paradiseus',
        'Puntius lateristriga', 'Setipinna taty', 'Sillaginopsis panijus'
    ]  # Ganti dengan nama kelas ikan yang sesuai
    predicted_class = class_names[predicted_class_index]

    return predicted_class

# Load model dan scaler
model_fish = load_model('supervised/model_fish.h5')  # Ganti dengan path model neural network Anda
scaler_fish = load_scaler_and_encoder('supervised/scaler_fish.pkl')  # Ganti dengan path scaler Anda

# App title
st.title("Aplikasi Prediksi Species Ikan")
st.subheader("Masukkan Input untuk Prediksi Species Ikan")

# Input user untuk ikan
length = st.number_input("Length (cm)", min_value=0.0, step=0.1)
weight = st.number_input("Weight (grams)", min_value=0.0, step=0.1)
w_l_ratio = st.number_input("Weight-to-Length Ratio", min_value=0.0, step=0.1)

# Susun input data untuk ikan
input_data = np.array([length, weight, w_l_ratio])

# Tombol prediksi
if st.button("Prediksi Species Ikan"):
    try:
        # Prediksi berdasarkan input
        predicted_fish = predict_fish(model_fish, scaler_fish, input_data)

        # Tampilkan hasil prediksi
        st.success(f"Prediksi Species: {predicted_fish}")
    except Exception as e:
        # Tangani kesalahan
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
