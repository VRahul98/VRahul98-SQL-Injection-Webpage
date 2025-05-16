import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
@st.cache
def load_data():
    data = pd.read_csv("Modified_SQL_Dataset.csv", encoding='ISO-8859-1')  # Replace "sql_injection_data.csv" with your dataset
    return data

# Train model
@st.cache
def train_model(data):
    # Split data into features and labels
    X = data['Query']
    y = data['Label']
    
    # Vectorize text data
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save the vectorizer and model
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(model, "model.pkl")
    
    # Evaluate model
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    return model, train_accuracy, test_accuracy

# Main function
def main():
    st.title("SQL Query Injection Prediction")
    
    # Load data
    data = load_data()
    
    # Train model
    model, train_accuracy, test_accuracy = train_model(data)
    
    st.write("Training Accuracy:", train_accuracy)
    st.write("Test Accuracy:", test_accuracy)
    
    # User input
    query = st.text_input("Enter SQL Query:")
    
    if query:
        # Load vectorizer and model
        vectorizer = joblib.load("vectorizer.pkl")
        model = joblib.load("model.pkl")
        
        # Vectorize input query
        query_vectorized = vectorizer.transform([query])
        
        # Predict vulnerability
        prediction = model.predict(query_vectorized)
        prediction_prob = model.predict_proba(query_vectorized)
        
        st.write("Prediction:", prediction[0])
        st.write("Prediction Probability (Not Vulnerable, Vulnerable):", prediction_prob[0])
        
# Run the app
if __name__ == "__main__":
    main()
