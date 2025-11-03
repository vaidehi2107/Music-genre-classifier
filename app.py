import streamlit as st
import pandas as pd
import joblib

import os
import joblib
import gdown

MODEL_PATH = "stack_model.pkl"

ENCODER_PATH = "label_encoder_top15.pkl"

# Google Drive direct download link
MODEL_ID = "1TgRfakQqSRfkErRYYV6gNJMvjdp8ZXyE"  # replace with your actual file ID
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# Download model only if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

# Load the model and label encoder
model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Load model and correct label encoder
model = joblib.load("stack_model.pkl")
le = joblib.load("label_encoder_top15.pkl")

st.set_page_config(page_title="Music Genre Classification", page_icon="🎵", layout="wide")

st.title("🎶 Music Genre Classification using Stacking Ensemble")
st.write("This app predicts the **music genre** based on track features using an ensemble of XGBoost, LightGBM, and CatBoost models.")

# Sidebar for input type
option = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV"])

# Features (must match training order)
features = [
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
    'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature', 'log_duration',
    'energy_danceability', 'speech_acoustic', 'energy_to_loudness',
    'acoustic_to_instrumental', 'tempo_bin'
]

# Manual input
if option == "Manual Input":
    st.subheader("Enter Song Features")
    user_data = {f: st.number_input(f"Enter {f}", value=0.0) for f in features}
    input_df = pd.DataFrame([user_data])

# CSV upload
else:
    st.subheader("Upload a CSV file")
    uploaded_file = st.file_uploader("Upload your CSV with the same feature columns", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(input_df.head())
    else:
        input_df = None

# Predict button
if st.button("🎧 Predict Genre"):
    if input_df is not None:
        preds = model.predict(input_df)
        decoded_preds = le.inverse_transform(preds)
        st.success(f"🎵 Predicted Genre: **{decoded_preds[0]}**")
    else:
        st.warning("Please provide input data first!")

st.markdown("---")
st.caption("Built by Vaidehi Dave • Machine Learning Coursework Project")

