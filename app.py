import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- 1. Data Handler (LLD Section 2) ---
class DataHandler:
    @staticmethod
    def load_data():
        [cite_start]"""Load or generate the dataset[cite: 10, 41]."""
        # [cite_start]Generating a simple 1D dataset as per Technical Specs [cite: 49]
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10 
        y = 2.5 * X + np.random.randn(100, 1) * 2
        return X, y

# --- 2. Regression Model (LLD Section 2) ---
class RegressionModel:
    def __init__(self):
        [cite_start]self.model = LinearRegression() # [cite: 71, 112]
    
    def train(self, X, y):
        [cite_start]self.model.fit(X, y) # [cite: 19, 75]
        
    def predict(self, x_input):
        [cite_start]return self.model.predict([[x_input]]) # [cite: 21, 79]

# --- 3. Visualizer (LLD Section 2) ---
class Visualizer:
    @staticmethod
    def plot_all(X, y, model, x_input=None, y_pred=None):
        fig, ax = plt.subplots()
        [cite_start]ax.scatter(X, y, color='blue', label='Data Points') # [cite: 41]
        
        # [cite_start]Regression Line [cite: 41, 53]
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        ax.plot(x_range, model.model.predict(x_range), color='red', label='Regression Line')
        
        # [cite_start]Highlight User Prediction [cite: 60]
        if x_input is not None:
            ax.scatter(x_input, y_pred, color='green', s=100, label='Your Prediction')
            
        ax.set_xlabel('X (Input)')
        ax.set_ylabel('Y (Target)')
        ax.legend()
        return fig

# --- 4. Streamlit App Script (LLD Section 2) ---
def main():
    [cite_start]st.title("Regressio - Linear Regression Web Demo") # [cite: 77]
    [cite_start]st.write("An interactive demo for learning linear regression concepts.") # [cite: 35, 42]

    # Initialize Components
    dh = DataHandler()
    X, y = dh.load_data()
    
    rm = RegressionModel()
    rm.train(X, y)
    
    # [cite_start]UI Layout [cite: 51-55]
    [cite_start]x_input = st.number_input("Enter X value:", value=5.0) # [cite: 79]
    y_pred = rm.predict(x_input)[0][0]
    
    [cite_start]st.write(f"### Predicted Y: {y_pred:.2f}") # [cite: 80]
    
    # [cite_start]Visualization [cite: 106]
    fig = Visualizer.plot_all(X, y, rm, x_input, y_pred)
    [cite_start]st.pyplot(fig) # [cite: 78]
    
    # [cite_start]Model Parameters [cite: 41, 84]
    st.subheader("Model Parameters")
    st.write(f"Slope: {rm.model.coef_[0][0]:.2f}")
    st.write(f"Intercept: {rm.model.intercept_[0]:.2f}")
    st.write(f"R² Score: {rm.model.score(X, y):.2f}")

if __name__ == "__main__":
    main()
