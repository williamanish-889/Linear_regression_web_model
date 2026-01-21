import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Data Handler ---
class DataHandler:
    @staticmethod
    def load_model(file_path):
        """Loads the pre-trained joblib model."""
        try:
            return joblib.load(file_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None


# --- 2. Regression Engine ---
class RegressionEngine:
    def __init__(self, model):
        self.model = model

    def predict(self, study_hours):
        """Predicts score based on study hours."""
        input_data = np.array([[study_hours]])  # Reshape to 2D
        return self.model.predict(input_data)[0]


# --- 3. Streamlit App ---
def main():
    st.set_page_config(page_title="Regressio - Study Score Predictor")

    st.title("Regressio - Linear Regression Web Demo")
    st.write(
        "Enter your study hours below to predict your test score "
        "based on the trained Linear Regression model."
    )

    # Load model
    model = DataHandler.load_model("linear_regression_model.pkl")

    st.write("Model loaded:", model is not None)

    if model is None:
        st.stop()

    engine = RegressionEngine(model)

    study_hours = st.number_input(
        "Enter Study Hours:",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    if st.button("Predict Score"):
        prediction = engine.predict(study_hours)
        st.success(f"Predicted Test Score: {prediction:.2f}")
            
        
            st.subheader("Prediction Graph")

            # Generate range of study hours
            x_range = np.linspace(0, 24, 100).reshape(-1, 1)
            y_range = model.predict(x_range)

            # Plot
            fig, ax = plt.subplots()
            ax.plot(x_range, y_range, label="Regression Line")
            ax.scatter(study_hours, prediction, color="red", label="Your Prediction", s=100)

            ax.set_xlabel("Study Hours")
            ax.set_ylabel("Predicted Score")
            ax.set_title("Study Hours vs Predicted Score")
            ax.legend()

            st.pyplot(fig)

            # Model insights
            st.subheader("Model Insights")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Model Slope (Coefficient)", f"{model.coef_[0]:.2f}")

            with col2:
                st.metric("Model Intercept", f"{model.intercept_:.2f}")

        # Explanation
        with st.expander("How does this work?"):
            st.write(
                "This app uses a Simple Linear Regression model trained "
                "to predict exam scores based on study hours."
            )
            st.latex(r"Y = mX + c")


if __name__ == "__main__":
    main()

