import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIGURATION ---
[cite_start]st.set_page_config(page_title="Regressio - Linear Regression Demo", layout="wide") [cite: 42, 44]

# [cite_start]--- 1. DATA HANDLER CLASS --- [cite: 10, 11]
class DataHandler:
    @staticmethod
    def load_data():
        [cite_start]"""Generates simple synthetic dataset for educational purposes.""" [cite: 41, 49]
        np.random.seed(42)
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X + np.random.randn(100, 1)
        df = pd.DataFrame(np.hstack([X, y]), columns=['X', 'y'])
        return df

# [cite_start]--- 2. REGRESSION MODEL CLASS --- [cite: 10, 104]
class RegressionModel:
    def __init__(self):
        [cite_start]self.model = LinearRegression() [cite: 6, 112]
        self.slope = 0
        self.intercept = 0
        self.r2 = 0

    def train(self, X, y):
        [cite_start]"""Fits the linear regression model and stores parameters.""" [cite: 10, 41]
        self.model.fit(X, y)
        self.slope = self.model.coef_[0][0]
        self.intercept = self.model.intercept_[0]
        self.r2 = self.model.score(X, y)

    def predict(self, x_value):
        [cite_start]"""Predicts Y based on a single X input.""" [cite: 10, 41]
        return self.model.predict([[x_value]])[0][0]

# [cite_start]--- 3. VISUALIZER CLASS --- [cite: 10, 104]
class Visualizer:
    @staticmethod
    def plot_regression(df, model, user_x=None, user_y=None):
        [cite_start]"""Renders scatter plot and regression line.""" [cite: 41, 58]
        fig, ax = plt.subplots()
        ax.scatter(df['X'], df['y'], color='blue', label='Data Points', alpha=0.5)
        
        # Regression line
        x_range = np.linspace(df['X'].min(), df['X'].max(), 100).reshape(-1, 1)
        y_range = model.model.predict(x_range)
        [cite_start]ax.plot(x_range, y_range, color='red', label='Regression Line') [cite: 41, 53]
        
        # Highlight user prediction
        if user_x is not None:
            [cite_start]ax.scatter(user_x, user_y, color='green', s=100, edgecolors='black', label='Your Prediction') [cite: 41, 60]
            
        [cite_start]ax.set_xlabel('Study Hours (X)') [cite: 59]
        [cite_start]ax.set_ylabel('Test Score (Y)') [cite: 59]
        ax.legend()
        return fig

# [cite_start]--- 4. MAIN APP ORCHESTRATION --- [cite: 10, 14]
def main():
    [cite_start]st.title("Regressio - Linear Regression Web Demo") [cite: 32, 77]
    [cite_start]st.markdown("An educational tool to understand the relationship between variables.") [cite: 35, 99]

    # Initialize data and model
    dh = DataHandler()
    df = dh.load_data()
    X = df[['X']]
    y = df['y']
    
    rm = RegressionModel()
    rm.train(X, y)

    # [cite_start]Sidebar / User Input [cite: 54, 104]
    st.sidebar.header("User Input")
    user_input = st.sidebar.number_input("Enter X value to predict Y:", 
                                         min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    # [cite_start]Predictions [cite: 106]
    prediction = rm.predict(user_input)

    # [cite_start]Layout Columns [cite: 51, 61]
    col1, col2 = st.columns([2, 1])

    with col1:
        [cite_start]st.subheader("Data Visualization") [cite: 8, 37]
        fig = Visualizer.plot_regression(df, rm, user_input, prediction)
        [cite_start]st.pyplot(fig) [cite: 78, 112]

    with col2:
        [cite_start]st.subheader("Results & Parameters") [cite: 41, 56]
        [cite_start]st.success(f"**Predicted Y Value:** {prediction:.2f}") [cite: 55, 80]
        
        [cite_start]st.write(f"**Slope (Coefficient):** {rm.slope:.2f}") [cite: 41, 84]
        [cite_start]st.write(f"**Intercept:** {rm.intercept:.2f}") [cite: 41, 84]
        [cite_start]st.write(f"**R² Score:** {rm.r2:.2f}") [cite: 41, 84]
        
        [cite_start]with st.expander("Educational Note"): [cite: 42, 57]
            st.write("Linear regression finds the 'best-fit' line by minimizing the sum of squared differences between observed and predicted values.")

if __name__ == "__main__":
    [cite_start]main() [cite: 15]
