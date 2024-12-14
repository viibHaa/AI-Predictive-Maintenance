import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model = joblib.load('C:/Users/Vibha/AI-Predictive-Maintenance/model.pkl')

# Title and description
st.title("Predict Remaining Useful Life (RUL)")
st.markdown("""
This app predicts the **Remaining Useful Life (RUL)** of machinery based on sensor data.
You can input the sensor readings manually or use sliders for a more interactive experience.
""")

# Sidebar for user input options
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Choose Input Method:", ["Manual Entry", "Sliders"])

# User input section
if input_method == "Manual Entry":
    st.subheader("Enter Sensor Data (comma-separated):")
    user_input = st.text_input("Sensor Data (comma-separated):")
else:
    st.subheader("Adjust Sensor Values Using Sliders:")
    sensor_1 = st.slider("Sensor 1", 0.0, 1.0, 0.5)
    sensor_2 = st.slider("Sensor 2", 0.0, 1.0, 0.5)
    sensor_3 = st.slider("Sensor 3", 0.0, 1.0, 0.5)
    sensor_4 = st.slider("Sensor 4", 0.0, 1.0, 0.5)
    # Add more sliders as needed
    slider_input = [sensor_1, sensor_2, sensor_3, sensor_4]  # Update with all sliders

# Predict button
if st.button("Predict"):
    try:
        # Prepare input data
        if input_method == "Manual Entry":
            input_features = np.array([float(x.strip()) for x in user_input.split(",")]).reshape(1, -1)
        else:
            input_features = np.array(slider_input).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)
        st.success(f"Predicted Remaining Useful Life (RUL): {prediction[0]:.2f}")

        # Visualization: Predicted RUL vs. Threshold
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["Predicted RUL"], [prediction[0]], color='blue', label='Predicted RUL')
        ax.axhline(y=100, color='red', linestyle='--', label='Threshold (100)')
        ax.set_ylim(0, max(150, prediction[0] + 20))
        ax.set_ylabel("RUL")
        ax.legend()
        st.pyplot(fig)

    except ValueError:
        st.error("Invalid input! Please ensure all values are numeric and comma-separated.")

# Optional Data Visualization
st.sidebar.subheader("Visualize Historical Data")
if st.sidebar.checkbox("Show Sample Data Distribution"):
    # Generate random data for visualization (replace with real data if available)
    data = np.random.normal(100, 15, 1000)  # Replace with actual RUL data
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data, kde=True, color='green', ax=ax)
    ax.set_title("Distribution of Remaining Useful Life (RUL)")
    ax.set_xlabel("RUL")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
