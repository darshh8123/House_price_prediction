import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Load the trained model and the feature list
try:
    model = joblib.load('bangalore_model.pkl')
    model_features = joblib.load('model_features.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run 'train_model.py' first.")
    st.stop()

# Get the list of locations from the features
locations = [col.replace('location_', '') for col in model_features if col.startswith('location_')]
locations.sort()

# Streamlit App Layout
st.set_page_config(page_title="Bangalore House Price Predictor", layout="centered")
st.title("Bangalore House Price Predictor")
st.markdown("---")

st.header("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    selected_location = st.selectbox('Select Location:', ['Select a location'] + locations)

with col2:
    bhk_size = st.selectbox('Number of BHK:', [1, 2, 3, 4])

area_input = st.slider('Total Area (sq. ft.):', min_value=200, max_value=10000, value=1200, step=100)

st.markdown("---")

if st.button("Predict Price", type="primary", use_container_width=True):
    if selected_location == 'Select a location':
        st.warning("Please select a location.")
    else:
        input_data = pd.DataFrame(columns=model_features, data=np.zeros((1, len(model_features))))
        input_data['total_sqft'] = area_input
        input_data['size'] = bhk_size
        
        location_col = f'location_{selected_location}'
        if location_col in input_data.columns:
            input_data[location_col] = 1

            predicted_price = model.predict(input_data)[0]

            st.success(f"### Predicted Price: ₹{predicted_price:,.2f}")
            st.balloons()
        else:
            st.error("Selected location not found in model features. Please try another location.")