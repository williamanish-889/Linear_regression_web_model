import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Data Handler (LLD Section 2) ---
class DataHandler:
    @staticmethod
    def load_model(file_path):
        [cite_start]"""Loads the pre-trained joblib model[cite: 10]."""
        try:
            return joblib.load(file_path)
        except Exception as e:
            [cite_start]st.error(f"Error loading model: {e} ")
            return None

# --- 2. Regression Engine (LLD Section 1) ---
class RegressionEngine:
    def __init__(self, model):
        self.model = model

    def predict(self, study_hours):
        [cite_start]"""Predicts score based on user input[cite: 10, 41]."""
        # [cite_start]Reshaping to 2D array as required by scikit-learn [cite: 12]
        input_data = np.array([[study_hours]])
        return self.model.predict(input_data)[0]

# --- 3. Streamlit App Orchestration (LLD Section 2) ---
def main():
    st.set_page_config(page_title="Regressio - Study Score Predictor")
    
    # [cite_start]UI Title and Description [cite: 52, 77]
    st.title("Regressio - Linear Regression Web Demo")
    [cite_start]st.write("Enter your study hours below to predict your test score based on the pre-trained model[cite: 35, 99].")
    
    # Load Model
    model = DataHandler.load_model('linear_regression_model.pkl')
    
    if model:
        engine = RegressionEngine(model)
        
        # [cite_start]User Input Field 
        study_hours = st.number_input(
            "Enter Study Hours:", 
            min_value=0.0, 
            max_value=24.0, 
            value=5.0, 
            step=0.5
        )
        
        # [cite_start]Prediction Logic [cite: 21, 106]
        if st.button("Predict Score"):
            prediction = engine.predict(study_hours)
            
            # [cite_start]Display Result [cite: 24, 55]
            st.success(f"### Predicted Test Score: {prediction:.2f}")
            
            # [cite_start]Show Model Parameters from the loaded pickle [cite: 41, 63]
            st.subheader("Model Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Slope (Coefficient)", f"{model.coef_[0]:.2f}")
            with col2:
                st.metric("Model Intercept", f"{model.intercept_:.2f}")

        # [cite_start]Educational Explanation [cite: 42, 57]
        with st.expander("How does this work?"):
            [cite_start]st.write("This app uses a Simple Linear Regression model trained on study hour data[cite: 6, 65].")
            st.write("Formula: $Y = (Slope \times X) + Intercept$")

if __name__ == "__main__":
    main()
