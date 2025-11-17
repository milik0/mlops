import streamlit as st
import joblib

model = joblib.load("regression.joblib")

st.title("House Price Prediction")

size = st.number_input("Size (in square feet)", min_value=0.0, value=1000.0, step=10.0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=2, step=1)
garden = st.number_input("Has Garden (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0, step=1)

if st.button("Predict Price"):
    input_data = [[size, bedrooms, garden]]
    prediction = model.predict(input_data)
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")