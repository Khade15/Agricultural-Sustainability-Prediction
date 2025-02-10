import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd

# Load the trained model
model = joblib.load('agricultural_sustainability_model_balanced.pkl')

# Streamlit UI
st.title("🌱 Agricultural Sustainability Prediction")
st.write("Enter farm parameters to predict sustainability.")

# Input fields
soil_health = st.slider("Soil Health", min_value=0.0, max_value=1.0, step=0.01)
crop_yield = st.number_input("Crop Yield (kg per hectare)", min_value=1000.0, max_value=10000.0, step=100.0)
water_usage = st.number_input("Water Usage (liters)", min_value=500.0, max_value=5000.0, step=100.0)
carbon_footprint = st.number_input("Carbon Footprint (kg CO2)", min_value=50.0, max_value=300.0, step=1.0)
fertilizer_use = st.number_input("Fertilizer Use (kg per hectare)", min_value=50.0, max_value=300.0, step=1.0)

# Predict button
if st.button("Predict Sustainability"):
    # Prepare input data
    input_data = np.array([[soil_health, crop_yield, water_usage, carbon_footprint, fertilizer_use]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.success("✅ The farm is *Sustainable*.")
    else:
        st.error("❌ The farm is *Unsustainable*.")

    # Show feature importance (optional)
    feature_names = ["Soil Health", "Crop Yield", "Water Usage", "Carbon Footprint", "Fertilizer Use"]
    feature_importance = model.feature_importances_

    df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    fig = px.bar(df, x="Feature", y="Importance", title="Feature Importance", color="Feature")
    st.plotly_chart(fig)