import streamlit as st
import pandas as pd
import joblib
import os
import sys

# -----------------------------
# Import path
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

st.title("🎓 Student Dropout Prediction System",text_alignment="center")
st.subheader("🔮 Predict student dropout risk using ML",text_alignment="center")

# -----------------------------
# INPUT UI
# -----------------------------
st.header("🧠 Mental Health")

col1, col2 = st.columns(2)

with col1:
    stress_level = st.slider("Stress Level", 0, 10, 5)
    anxiety_score = st.slider("Anxiety Score", 0, 10, 5)
    depression_score = st.slider("Depression Score", 0, 10, 5)

with col2:
    burnout_score = st.slider("Burnout Score", 0, 10, 5)
    mental_health_index = st.slider("Mental Health Index", 0, 100, 50)

# -----------------------------
st.header("📚 Academic")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Study Hours per Day", 0, 12, 4)
    academic_performance = st.slider("Academic Performance", 0, 100, 60)

with col2:
    exam_pressure = st.slider("Exam Pressure", 0, 10, 5)

# -----------------------------
st.header("💰 Lifestyle & Social")

col1, col2 = st.columns(2)

with col1:
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
    physical_activity = st.slider("Physical Activity", 0, 10, 5)
    financial_stress = st.slider("Financial Stress", 0, 10, 5)

with col2:
    family_expectation = st.slider("Family Expectation", 0, 10, 5)
    social_support = st.slider("Social Support", 0, 10, 5)

# -----------------------------
st.header("👤 Basic Info")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 15, 30, 20)

with col2:
    risk_level = st.selectbox("Initial Risk Level", ["Low", "Medium", "High"])
    academic_year = st.selectbox("Academic Year", [1, 2, 3, 4])

# -----------------------------
st.header("📱 Digital Usage")

col1, col2 = st.columns(2)

with col1:
    screen_time = st.slider("Screen Time (hrs/day)", 0, 12, 5)

with col2:
    internet_usage = st.slider("Internet Usage (hrs/day)", 0, 12, 5)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Predict Dropout Risk"):

    input_data = pd.DataFrame({
        "stress_level": [stress_level],
        "burnout_score": [burnout_score],
        "depression_score": [depression_score],
        "anxiety_score": [anxiety_score],
        "mental_health_index": [mental_health_index],
        "sleep_hours": [sleep_hours],
        "physical_activity": [physical_activity],
        "study_hours_per_day": [study_hours],
        "academic_performance": [academic_performance],
        "exam_pressure": [exam_pressure],
        "financial_stress": [financial_stress],
        "family_expectation": [family_expectation],
        "social_support": [social_support],
        "gender": [gender],
        "risk_level": [risk_level],
        "age": [age],
        "academic_year": [academic_year],
        "screen_time": [screen_time],
        "internet_usage": [internet_usage]
    })

    # Prediction
    pred = model.predict(input_data)
    pred_label = le.inverse_transform(pred)[0]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("📊 Result")

    if pred_label == "High":
        st.error("⚠️ High Dropout Risk")
    elif pred_label == "Medium":
        st.warning("⚠️ Medium Dropout Risk")
    else:
        st.success("✅ Low Dropout Risk")