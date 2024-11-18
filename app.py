import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv(r"./diabetes_prediction_dataset.csv")
data["gender"] = data["gender"].map({"Male": 1, "Female": 2, "Other": 3})
data["smoking_history"] = data["smoking_history"].map({
    "never": 1, "No Info": 2, "current": 3, "former": 4, "ever": 5, "not current": 6
})

# Split data into training and testing sets
y = data['diabetes']
x = data.drop("diabetes", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a RandomForest model
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 5px;
        border: none;
        font-size: 1rem;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

# Application title
st.title("🩺 Diabetes Risk Prediction")

# Input form layout
with st.form("diabetes_form"):

    # Add some space
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Create a 2-column layout
    spacer, col1, spacer, col2, spacer = st.columns([0.2, 1, 0.1, 1, 0.2])

    # Input fields in the left column
    with col1:
        age = st.slider("Age", 0, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        hypertension = st.selectbox("Hypertension", [0, 1], help="0: No, 1: Yes")
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)

    # Input fields in the right column
    with col2:
        smoking_history = st.selectbox("Smoking History", ["Never", "No Info", "Current", "Former", "Ever", "Not Current"])
        heart_disease = st.selectbox("Heart Disease", [0, 1], help="0: No, 1: Yes")
        hba1c = st.slider("HbA1c Level", 4.0, 15.0, 5.5)
        blood_glucose = st.slider("Blood Glucose Level", 50, 250, 100)

    # Submit button
    submit_button = st.form_submit_button(label="🔍 Predict Risk")

# Handle prediction on form submission
if submit_button:
    gender_map = {"Male": 1, "Female": 2, "Other": 3}
    smoking_map = {"Never": 1, "No Info": 2, "Current": 3, "Former": 4, "Ever": 5, "Not Current": 6}
    input_data = np.array([[age, gender_map[gender], smoking_map[smoking_history], hypertension, heart_disease, bmi, hba1c, blood_glucose]])

    # Predict using the model
    prediction = rf_model.predict(input_data)
    result = "High risk of diabetes." if prediction[0] == 1 else "Low risk of diabetes."
    
    # Display the result
    result_color = 'red' if prediction[0] == 1 else 'green'
    st.markdown(f"<h3 style='text-align: center; color: {result_color};'>{result}</h3>", unsafe_allow_html=True)

# Display feature importance with updated style
st.subheader("🔍 Feature Importance Analysis")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = x.columns

# Plot feature importance
plt.figure(figsize=(10, 7))
plt.title("Feature Importance", fontsize=18, color="#2c3e50")
plt.bar(range(x.shape[1]), importances[indices], align="center", color="#2980b9", edgecolor="#2c3e50")
plt.xticks(range(x.shape[1]), features[indices], rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)
