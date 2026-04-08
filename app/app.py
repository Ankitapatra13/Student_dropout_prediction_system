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
DATA_PATH = os.path.join(BASE_DIR,"models","final_dataset.csv") 
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
st.divider()
# -----------------------------
# INPUT UI
# -----------------------------
st.header("🧠 Mental Health",text_alignment="center")

col1, col2 = st.columns(2)

with col1:
    stress_level = st.slider("Stress Level", 0, 10, 5)
    anxiety_score = st.slider("Anxiety Score", 0, 10, 5)
    depression_score = st.slider("Depression Score", 0, 10, 5)

with col2:
    burnout_score = st.slider("Burnout Score", 0, 10, 5)
    mental_health_index = st.slider("Mental Health Index", 0, 100, 50)
st.divider()
# -----------------------------
st.header("📚 Academic",text_alignment="center")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Study Hours per Day", 0, 12, 4)
    academic_performance = st.slider("Academic Performance", 0, 100, 60)

with col2:
    exam_pressure = st.slider("Exam Pressure", 0, 10, 5)
st.divider()
# -----------------------------
st.header("💰 Lifestyle & Social",text_alignment="center")

col1, col2 = st.columns(2)

with col1:
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
    physical_activity = st.slider("Physical Activity", 0, 10, 5)
    financial_stress = st.slider("Financial Stress", 0, 10, 5)

with col2:
    family_expectation = st.slider("Family Expectation", 0, 10, 5)
    social_support = st.slider("Social Support", 0, 10, 5)
st.divider()
# -----------------------------
st.header("👤 Basic Info",text_alignment="center")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 15, 30, 20)

with col2:
    risk_level = st.selectbox("Initial Risk Level", ["Low", "Medium", "High"])
    academic_year = st.selectbox("Academic Year", [1, 2, 3, 4])
st.divider()
# -----------------------------
st.header("📱 Digital Usage",text_alignment="center")

col1, col2 = st.columns(2)

with col1:
    screen_time = st.slider("Screen Time (hrs/day)", 0, 12, 5)

with col2:
    internet_usage = st.slider("Internet Usage (hrs/day)", 0, 12, 5)
st.divider()
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
    features = joblib.load(os.path.join(BASE_DIR,"models","features.pkl"))
    input_data = input_data[features]
    pred = model.predict(input_data)
    pred_label = le.inverse_transform(pred)[0]
    pred_proba = model.predict_proba(input_data)
    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("🔮 Prediction",text_alignment="center")
    col1,col2 = st.columns(2)
    with col1:

        if pred_label == "High":
            st.error("⚠️ High Dropout Risk")
        elif pred_label == "Medium":
            st.warning("⚠️ Medium Dropout Risk")
        else:
            st.success("✅ Low Dropout Risk")
        confidence = pred_proba.max()
        st.write(f"🎯 Confidence Score : **{confidence:.2f}** ")
    
    with col2:
        st.subheader("📊 Risk Distribution")
        proba_df = pd.DataFrame(pred_proba,columns=le.classes_)
        st.bar_chart(proba_df.T)
    
    st.divider()
    # -----------------------------
    # INSIGHTS
    # -----------------------------
    df = pd.read_csv(DATA_PATH)
    st.subheader("🧠 Key Insights",text_alignment="center")
    insights = []
    if stress_level > df["stress_level"].mean() :
        insights.append("High stress level is increasing dropout risk")
    if anxiety_score > df["anxiety_score"].mean() :
        insights.append("High anxiety impacts performance")
    if depression_score > df["depression_score"].mean():
        insights.append("High depression impacts performance")
    if burnout_score > df["burnout_score"].mean():
        insights.append("High burnout increases dropout risk")
    if study_hours < 4:
        insights.append("Low study hours impacts lack of interest in course")
    if sleep_hours < 6:
        insights.append("Low sleep reduces mental stability")
    if academic_performance < df["academic_performance"].mean():
        insights.append("Low academic performance is a major risk factor")
    if social_support < 3:
        insights.append("Low social support increases risk")
    if exam_pressure > df["exam_pressure"].mean():
        insights.append("High exam pressure impacts performance")
    if physical_activity < 3:
        insights.append("Low physical activity leads to poor mental health")
    if financial_stress > 7:
        insights.append("High financial stress increases dropout risk")
    if family_expectation > 7:
        insights.append("High family expectation increases stress")
    if mental_health_index < 30:
        insights.append("Low mental health increases the dropout risk")
    

    if not insights :
        insights.append("No major risk factors detected !")
    
    for i in insights :
        st.markdown(f"• {i}")
    
    st.divider()
    #----------------------------------
    # RECOMMENDATIONS
    #----------------------------------
    st.subheader("💡 Recommendations",text_alignment="center")
    recommendations = []

    if stress_level > df["stress_level"].mean() :
        recommendations.append("🧘 Practice stress management through meditation and yoga")
    if anxiety_score > df["anxiety_score"].mean() :
        recommendations.append("Consider mental health support or counseling for calming anxiety")
    if depression_score > df["depression_score"].mean():
        recommendations.append("Sharing suppressed emotions can releif depression")
    if burnout_score > df["burnout_score"].mean():
        recommendations.append("Take regular breaks to avoid burnout")
    if study_hours < 4:
        recommendations.append("🧑‍🎓 Join the course of your interest, to spend time in studies")
    if sleep_hours < 6:
        recommendations.append("🛌 Maintain 7-8 hours of healthy sleep per day")
    if academic_performance < df["academic_performance"].mean():
        recommendations.append("📚 Build interest in the course by putting sincere effort for the subject")
    if social_support < 3:
        recommendations.append("👥 Grow a healthy social circle to nourish knowledge")
    if exam_pressure > df["exam_pressure"].mean():
        recommendations.append("Practice mock tests to defeat exam fear")
    if physical_activity < 3:
        recommendations.append("🏋️ Consider joining gym, or start any physical activity to stay fit")
    if financial_stress > 7:
        recommendations.append("🧩 Start internships, provide tutions, work on a startup idea to start early income")
    if family_expectation > 7:
        recommendations.append("Manage expectations through structured planning and communication")
    if mental_health_index < 30:
        recommendations.append("🎨 Spending leisure in mindfull, joyous activities can cheer up mental health")
    if not recommendations :
        recommendations.append("✅ Maintain current healthy routine")
    
    for r in recommendations :
        st.markdown(f"✅ {r}")


