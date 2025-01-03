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

# Fungsi untuk prediksi buah
def predict_fruit(model, scaler, input_data):
    # Scaling data input sebelum prediksi
    input_data_scaled = scaler.transform([input_data])  # Pastikan input sudah sesuai dengan scaler

    # Prediksi kelas buah
    prediction = model.predict(input_data_scaled)
    
    # Ambil indeks kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Tentukan nama kelas berdasarkan indeks
    class_names = [
        'grapefruit','orange'
    ]  # Ganti dengan nama kelas buah yang sesuai
    predicted_class = class_names[predicted_class_index]

    return predicted_class

# Load model dan scaler
model_fruit = load_model('supervised/model_fruit.h5')  # Ganti dengan path model neural network Anda
scaler_fruit = load_scaler_and_encoder('supervised/scaler_fruit.pkl')  # Ganti dengan path scaler Anda

# App title
st.title("Aplikasi Prediksi Species Buah")
st.subheader("Masukkan Input untuk Prediksi Species Buah")

# Input user untuk buah
diameter = st.number_input("Diameter (cm)", min_value=0.0, step=0.1)
weight = st.number_input("Weight (grams)", min_value=0.0, step=0.1)
red = st.number_input("Red (RGB Value)", min_value=0.0, max_value=255.0, step=1.0)
green = st.number_input("Green (RGB Value)", min_value=0.0, max_value=255.0, step=1.0)
blue = st.number_input("Blue (RGB Value)", min_value=0.0, max_value=255.0, step=1.0)

# Susun input data untuk buah
input_data = np.array([diameter, weight, red, green, blue])

# Tombol prediksi
if st.button("Prediksi Species Buah"):
    try:
        # Prediksi berdasarkan input
        predicted_fruit = predict_fruit(model_fruit, scaler_fruit, input_data)

        # Tampilkan hasil prediksi
        st.success(f"Prediksi Species: {predicted_fruit}")
    except Exception as e:
        # Tangani kesalahan
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
