import numpy as np 
import pandas as pd
import joblib
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import streamlit as st

uri = "mongodb+srv://akash:akash@cluster0.44hv4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))


# Load models
xgboost_model = joblib.load('reg.pkl')
knn_model = joblib.load('knn_10.pkl')

db = client.models_db

def predict_xgboost(input_data):
    return xgboost_model.predict(input_data)

def predict_knn(input_data):
    return knn_model.predict(input_data)

def main():
    st.title("Energy Prediction and KNN Car Prediction")

    # Features for XGBoost model prediction
    st.header("XGBoost Energy Consumption Prediction")
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, step=1)
    dayofweek = st.number_input("Day of Week (0-6)", min_value=0, max_value=6, step=1)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, step=1)
    year = st.number_input("Year (e.g., 2021)", min_value=2000, max_value=2025, step=1)

    # Calculate day of the year based on the selected year and month
    dayofyear = (pd.to_datetime(f"{year}-{month}-01") + pd.DateOffset(months=1) - pd.DateOffset(days=1)).day

    if st.button("Predict Energy Consumption"):
        # Create input DataFrame with the correct feature names expected by the model
        input_data = pd.DataFrame({
            'hour': [hour],
            'dayofweek': [dayofweek],
            'month': [month],
            'year': [year],
            'dayofyear': [dayofyear]
        })
        
        # Make sure correct types
        input_data = input_data.astype({'hour': 'int', 'dayofweek': 'int', 'month': 'int', 'year': 'int', 'dayofyear': 'int'})

        # Try to get prediction
        try:
            prediction = predict_xgboost(input_data)
            predicted_value = float(prediction[0]) 
            st.success(f'Predicted Energy Consumption: {prediction[0]:.2f} MW')
            
            # Save to MongoDB
            db.xgboost_model.insert_one({
                "hour": hour,
                "dayofweek": dayofweek,
                "month": month,
                "year": year,
                "dayofyear": dayofyear,
                "prediction": predicted_value
            })
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Features for KNN model prediction
    st.header("KNN Car Specifications Prediction")
    horsepower = st.number_input("Horsepower", min_value=0.0, step=0.1)
    weight = st.number_input("Weight", min_value=0.0, step=0.1)
    mpg = st.number_input("MPG", min_value=0.0, step=0.1)  
    
    if st.button("Predict Car Displacement"):
        input_data_knn = pd.DataFrame({
            'horsepower': [horsepower],
            'weight': [weight],
            'mpg':[mpg]
        })
        
        # Ensure correct types
        input_data_knn = input_data_knn.astype({'horsepower': 'float', 'weight': 'float','mpg':'float'})
        
        predicted_displacement = predict_knn(input_data_knn)
        st.success(f'Predicted Car Displacement: {predicted_displacement[0]:.2f}')
        
        # Optionally save to MongoDB
        db.knn_model.insert_one({
            "horsepower": horsepower,
            "weight": weight,
            "mpg":mpg,
            "predicted_displacement": float(predicted_displacement[0])
        })

if __name__ == "__main__":
    main()