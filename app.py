import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- 1. Data Handler Class (LLD 2.0) ---
class DataHandler:
    @staticmethod
    def load_data():
        """Load or generate the dataset."""
        # Simple 1D synthetic dataset as per PRD FR-1
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        y = 2.5 * X + np.random.randn(100, 1) * 2
        return X, y

# --- 2. Regression Model Class (LLD 2.0) ---
class RegressionModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, X, y):
        """Fits the model using scikit-learn."""
        self.model.fit(X, y)
        
    def predict(self, x_input):
        """Returns the prediction for a given X."""
        return self.model.predict([[x_input]])

# --- 3. Visualizer Class (LLD 2.0) ---
class Visualizer:
    @staticmethod
    def plot_all(X, y, rm, x_input=None, y_pred=None):
        """Generates scatter plot and regression line."""
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label='Actual Data', alpha=0.6)
        
        # Regression Line calculation
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_range = rm.model.predict(x_range)
        ax.plot(x_range, y_range, color='red', linewidth=2, label='Regression Line')
        
        # Highlight User Prediction
        if x_input is not None:
            ax.scatter(x_input, y_pred, color='green', s=150, edgecolors='black', label='Your Prediction')
            
        ax.set_xlabel('X Value')
        ax.set_ylabel('Y Value')
        ax.legend()
        return fig

# --- 4. Main App Orchestration ---
def main():
    st.set_page_config(page_title="Regressio Demo")
    st.title("Regressio - Linear Regression Web Demo")
    st.write("This tool demonstrates simple linear regression concepts.")

    # Data & Model Setup
    dh = DataHandler()
    X, y = dh.load_data()
    
    rm = RegressionModel()
    rm.train(X, y)
    
    # Sidebar for User Interaction (PRD FR-5)
    st.sidebar.header("Input Parameters")
    x_val = st.sidebar.number_input("Enter an X value to predict Y:", value=5.0)
    
    # Prediction (PRD FR-6)
    prediction = rm.predict(x_val)[0][0]
    
    # UI Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Visualization")
        fig = Visualizer.plot_all(X, y, rm, x_val, prediction)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Prediction Result")
        st.metric(label="Predicted Y", value=f"{prediction:.2f}")
        
        st.subheader("Model Parameters")
        st.write(f"**Slope (m):** {rm.model.coef_[0][0]:.2f}")
        st.write(f"**Intercept (b):** {rm.model.intercept_[0]:.2f}")
        st.write(f"**R² Score:** {rm.model.score(X, y):.2f}")

if __name__ == "__main__":
    main()
