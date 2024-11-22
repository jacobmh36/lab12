import streamlit as st
import pickle
import numpy as np

# Load the trained model
import os
import requests
import streamlit as st

# Link to the external model file
MODEL_URL = "https://drive.google.com/uc?id=1wCUJl6ItEfAw564cYa53TA6jc7pHdQkKE"  # Replace with your Google Drive file's direct link

# Download the model if not already present
MODEL_PATH = "regression_model.pkl"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading the model..."):
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download the model. Please check the link or try again.")

# Load the trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# App title
st.title("California Housing Price Predictor")

st.write(
    """
    This app predicts **California Housing Prices** based on several features.
    Adjust the sliders below to input feature values and see the predicted price.
    """
)

# User input for features using sliders
MedInc = st.slider("Median Income", 0.0, 150000.0, 50000.0)
HouseAge = st.slider("House Age", 0, 52, 25)
AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.slider("Population", 0, 40000, 1500)
AveOccup = st.slider("Average Occupancy", 0.5, 10.0, 3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 37.0)
Longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

# Prepare the input for prediction
input_data = np.array([[MedInc/10000, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Prediction
prediction = model.predict(input_data)
st.subheader("Predicted Housing Price")
st.write(f"**${prediction[0]*100000:.2f}**")
