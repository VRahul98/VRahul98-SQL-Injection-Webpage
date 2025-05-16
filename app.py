import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("SQL Injection Prediction 🔐")

user_input = st.text_area("Enter your query or input string:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a query to predict.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        label = "SQL Injection ❌" if prediction == 1 else "Benign ✅"
        st.subheader(f"Prediction: {label}")
