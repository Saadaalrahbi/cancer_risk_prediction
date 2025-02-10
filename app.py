
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset for feature scaling
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Standardize features (fit on the training data)
scaler = StandardScaler()
scaler.fit(df)  # Fit scaler on the original dataset

# Train model (just an example of how you can fit the model)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(scaler.transform(df), data.target)  # Use scaled data to train

# Function to make predictions
def predict_cancer_risk(model, patient_data):
    """
Predicts cancer risk for a new patient.
patient_data: List of 30 numerical values corresponding to features.
    """
    patient_data = np.array(patient_data).reshape(1, -1)  # Ensure it's in the right shape
    patient_data_scaled = scaler.transform(patient_data)  # Apply scaling
    prediction = model.predict(patient_data_scaled)[0]
    
    if prediction == 0:
        return "High Cancer Risk (Malignant)"
    else:
        return "Low Cancer Risk (Benign)"


# Streamlit interface
st.title("Breast Cancer Prediction")

# User input form
input_data = []
for feature in df.columns:
    value = st.number_input(f"Enter value for {feature}", min_value=0.0, max_value=100.0, step=0.1)
    input_data.append(value)

# Prediction button
if st.button("Predict Cancer Risk"):
    prediction = predict_cancer_risk(model, input_data)
    st.write(f"**Prediction: {prediction}**")
