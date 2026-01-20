
import streamlit as st
import pandas as pd
import joblib
import os

# 1. Load the pre-trained Linear Regression model
# Ensure the model file exists in the same directory or provide the correct path
try:
    model = joblib.load('linear_regression_model.pkl')
    # print("Model loaded successfully.") # This would print to the console where Streamlit is run
except FileNotFoundError:
    st.error("Error: 'linear_regression_model.pkl' not found. Please ensure the model is saved.")
    st.stop() # Stop the app if model is not found

# 2. Load model performance metrics
try:
    with open('model_metrics.txt', 'r') as f:
        metrics = f.readlines()
    r_squared = metrics[0].split(': ')[1].strip()
    mae = metrics[1].split(': ')[1].strip()
except FileNotFoundError:
    r_squared = "N/A"
    mae = "N/A"
    st.warning("Warning: 'model_metrics.txt' not found. Model performance metrics will not be displayed.")

# Set up the Streamlit application title and description
st.title("Linear Regression Model Predictor")
st.write("Enter the study hours to predict the test score.")

# Create an input widget for 'Study_Hours'
study_hours = st.number_input(
    label='Enter Study Hours',
    min_value=0.0,
    max_value=24.0,
    value=10.0,
    step=0.1
)

# Create a button for making predictions
if st.button('Predict Test Score'):
    # Reshape the input for the model
    input_data = pd.DataFrame([[study_hours]], columns=['Study_Hours'])
    
    # Make prediction
    predicted_score = model.predict(input_data)[0]
    
    # Display the predicted 'Test_Score'
    st.success(f"Predicted Test Score: {predicted_score:.2f}")

# Display model performance metrics
st.subheader("Model Performance Metrics (on Test Set)")
st.metric(label="R-squared", value=r_squared)
st.metric(label="Mean Absolute Error (MAE)", value=mae)

# Add a feature to allow users to download the original data.csv file
st.subheader("Download Data")
if os.path.exists('data.csv'):
    with open('data.csv', 'rb') as f:
        st.download_button(
            label="Download Original data.csv",
            data=f,
            file_name="data.csv",
            mime="text/csv"
        )
else:
    st.warning("Original 'data.csv' not found for download.")
