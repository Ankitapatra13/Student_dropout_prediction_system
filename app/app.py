import streamlit as st
import pandas as pd
import joblib

from src.features import create_features

model = joblib.load("models/model.pkl")
le = joblib.load("models/label_encoder.pkl")

st.title("🎓 Student Dropout Risk Prediction")

# Example inputs (you can expand)
gender = st.selectbox("Gender", ["Male", "Female"])
stress = st.slider("Stress Level", 0, 10)
anxiety = st.slider("Anxiety Score", 0, 10)
mental = st.slider("Mental Health Index", 0, 10)

input_data = pd.DataFrame([{
    "gender": gender,
    "stress_level": stress,
    "anxiety_score": anxiety,
    "mental_health_index": mental,
}])

if st.button("Predict"):
    input_data = create_features(input_data)
    pred = model.predict(input_data)
    result = le.inverse_transform(pred)

    st.success(f"Predicted Risk: {result[0]}")