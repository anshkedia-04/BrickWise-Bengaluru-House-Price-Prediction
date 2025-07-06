import matplotlib.pyplot as plt
import streamlit as st
import json
import numpy as np
import joblib

# Load model and column data
model = joblib.load('bangalore_home_price_model.pkl')
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']
locations = data_columns[3:]  # assuming first 3 are sqft, bath, bhk

st.title("🏠 Bengaluru House Price Predictor")

# Input UI
sqft = st.number_input("Total Square Feet", min_value=300)
bhk = st.selectbox("BHK", [1,2,3,4,5])
bath = st.selectbox("Bathrooms", [1,2,3,4,5])
location = st.selectbox("Location", sorted(locations))

if st.button("Predict Price"):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location.lower() in data_columns:
        loc_index = data_columns.index(location.lower())
        x[loc_index] = 1

    prediction = model.predict([x])[0]
    st.success(f"Estimated Price: ₹ {round(prediction, 2)} Lakhs")


if st.sidebar.checkbox("Show Price Distribution"):
    # Example dummy chart
    prices = np.random.normal(70, 20, 1000)
    fig, ax = plt.subplots()
    ax.hist(prices, bins=30, color="skyblue")
    st.sidebar.pyplot(fig)
