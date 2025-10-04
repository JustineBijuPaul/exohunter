from fastapi import FastAPI
import streamlit as st
import numpy as np
import requests

def prediction_form():
    st.header("Exoplanet Classification Prediction")
    
    # Input fields for features
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)
    feature5 = st.number_input("Feature 5", value=0.0)
    feature6 = st.number_input("Feature 6", value=0.0)
    feature7 = st.number_input("Feature 7", value=0.0)
    feature8 = st.number_input("Feature 8", value=0.0)

    features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]

    if st.button("Predict"):
        # Call the FastAPI prediction endpoint
        response = requests.post("http://localhost:8000/predict", json={"features": features})
        
        if response.status_code == 200:
            predictions = response.json().get("predictions", {})
            st.success(f"Predictions: {predictions}")
        else:
            st.error("Error in prediction")