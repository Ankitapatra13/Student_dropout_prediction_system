# 🎓 Student Dropout Prediction System

## 📌 Overview
This project predicts **student dropout risk** (Low, Medium, High) using machine learning based on behavioral, academic, and mental health features.

The goal is to help identify at-risk students early and enable timely intervention.

---

## 🚀 Features
- 📊 End-to-end ML pipeline (preprocessing → training → prediction)
- 🧠 Feature engineering using psychological and academic indicators
- ⚖️ Class imbalance handling using sample weights
- 🤖 XGBoost & Logistic Regression model comparison
- 🌐 Deployed using Streamlit for real-time predictions

---

## 📊 Model Performance

| Model | Accuracy | F1 Score |
|------|--------|--------|
| Logistic Regression | 0.63 | 0.64 |
| XGBoost | 0.64 | 0.63 |
| XGBoost + Feature Engineering | 0.63 | 0.64 |

### 🔍 Key Insights
- Logistic Regression performed **on par with XGBoost**, indicating limited nonlinear separability
- Feature engineering gave **marginal improvements**, suggesting strong existing signal
- Medium class remains **challenging due to overlap**

---

## 📦 Dataset
- Original dataset size: **~20 million samples**
- Used subset: **10,000 samples**

### ⚠️ Note
A subset was used to:
- enable faster experimentation  
- perform efficient feature engineering  
- work within computational constraints  

---

## 🛠️ Tech Stack
- Python 🐍  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  

---


## 📁 Project Structure

```
Student_dropout_prediction_system/
│
├── app/                          # 🚀 Streamlit web application
│   └── app.py
│
├── data/                         # 📊 Dataset (sample used for training)
│   └── student_mental_health_dataset.csv
|
├── models/                       # 🤖 Saved trained models
|   │──model.pkl
|   │──label_encoder.pkl
|    
│   
|   

|
├── notebooks/                    # 📓 Jupyter notebooks (EDA & analysis)
│   └── 01_eda.ipynb
│
├── src/                          # 🧠 Core ML pipeline code
│   |── features.py               # Feature creation logic
|   |── preprocess.py             # Data cleaning & encoding
│   ├── train.py                  # Model training pipeline
│
│
├── .gitignore                    # 🚫 Ignored files
├── README.md                     # 📄 Project documentation
├── requirements.txt              # ⚙️ Project dependencies
|
```

> ⚠️ **Note:** The original dataset contains ~20 million records.  
> For computational efficiency, a sampled subset of 10,000 records was used for model development.
