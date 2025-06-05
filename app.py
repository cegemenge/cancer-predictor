
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("cancer-data-2.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.title("Breast Cancer Prediction App")
st.markdown("Enter measurements to predict if the tumor is benign (0) or malignant (1)")

# Input sliders
user_input = []
for col in X.columns:
    val = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    user_input.append(val)

# Prediction
if st.button("Predict"):
    prediction = model.predict([user_input])
    st.success(f"Prediction: {'Malignant (1)' if prediction[0] == 1 else 'Benign (0)'}")
