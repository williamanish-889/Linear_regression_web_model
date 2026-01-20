import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# [cite_start]1. Data Handler Module [cite: 10, 104]
class DataHandler:
    @staticmethod
    def load_data():
        # [cite_start]FR-1: Load simple dataset [cite: 41]
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        y = 2.5 * X + np.random.randn(100, 1) * 2
        return X, y

# [cite_start]2. Regression Engine Module [cite: 8, 10]
class RegressionModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, X, y):
        # [cite_start]FR-3: Fit linear regression model [cite: 41]
        self.model.fit(X, y)
        
    def predict(self, x_input):
        # [cite_start]FR-5: Predict Y from input [cite: 41]
        return self.model.predict([[x_input]])

# [cite_start]3. Visualization Engine [cite: 104]
class Visualizer:
    @staticmethod
    def plot_data(X, y, model, x_input, y_pred):
        # [cite_start]FR-2, FR-4, FR-6: Scatter plot and regression line [cite: 41]
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label='Actual Data', alpha=0.5)
        
        # Plot regression line
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        ax.plot(x_range, model.model.predict(x_range), color='red', label='Fit Line')
        
        # Highlight prediction
        ax.scatter(x_input, y_pred, color='green', s=100, label='Prediction')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        return fig

# [cite_start]4. Streamlit App Script [cite: 10]
def main():
    st.title("Regressio - Linear Regression Web Demo") # cite: 32, 77
    
    # Initialize logic
    dh = DataHandler()
    X, y = dh.load_data()
    
    rm = RegressionModel()
    rm.train(X, y)
    
    # [cite_start]User input [cite: 54, 79]
    x_val = st.number_input("Enter X value:", value=5.0)
    y_pred = rm.predict(x_val)[0][0]
    
    # [cite_start]Display results [cite: 55, 80]
    st.write(f"### Predicted Y: {y_pred:.2f}")
    
    # [cite_start]Visual Output [cite: 110]
    fig = Visualizer.plot_data(X, y, rm, x_val, y_pred)
    st.pyplot(fig)
    
    # [cite_start]Display Model Parameters [cite: 41, 84]
    st.subheader("Model Metrics")
    st.write(f"Slope: {rm.model.coef_[0][0]:.2f}")
    st.write(f"Intercept: {rm.model.intercept_[0]:.2f}")
    st.write(f"R² Score: {rm.model.score(X, y):.2f}")

if __name__ == "__main__":
    main()
