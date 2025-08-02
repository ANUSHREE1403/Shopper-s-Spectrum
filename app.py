import streamlit as st
import pandas as pd
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from datetime import datetime

# Load models
model = joblib.load('lightgbm_final_model.pkl')
encoder = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Shopper Spectrum Classifier", layout="wide")
st.title("🛍️ Shopper Spectrum Classification App")
st.markdown("Predict customer segments based on online purchase behavior.")

# Define input function
def user_input_features():
    # Basic product info
    description = st.text_input("Description", "WHITE HANGING HEART T-LIGHT HOLDER")
    quantity = st.number_input("Quantity", 0, 10000, 6)
    unitprice = st.number_input("Unit Price", 0.0, 1000.0, 2.55)
    
    # Date and time features
    st.subheader("Date and Time Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year = st.number_input("Year", 2010, 2025, 2010)
        month = st.number_input("Month", 1, 12, 12)
        day = st.number_input("Day", 1, 31, 1)
    
    with col2:
        hour = st.number_input("Hour", 0, 23, 14)
        dayofweek = st.number_input("Day of Week (0=Monday)", 0, 6, 0)
        weekday = st.number_input("Weekday (1=Weekday, 0=Weekend)", 0, 1, 1)
    
    with col3:
        is_weekend = st.selectbox("Is Weekend", [0, 1])
        is_night = st.selectbox("Is Night", [0, 1])
        country_encoded = st.selectbox("Country Encoded", [0, 1, 2, 3])
    
    # Calculate derived features
    total_price = quantity * unitprice
    log_quantity = np.log1p(quantity)
    log_unitprice = np.log1p(unitprice)
    log_total_price = np.log1p(total_price)

    # Use a placeholder numeric value for description since model expects it
    description_encoded = 0  # Placeholder - in real scenario this would be encoded

    data = {
        'description': description_encoded,
        'quantity': quantity,
        'unitprice': unitprice,
        'year': year,
        'month': month,
        'day': day,
        'hour': hour,
        'dayofweek': dayofweek,
        'total_price': total_price,
        'weekday': weekday,
        'country_encoded': country_encoded,
        'is_weekend': is_weekend,
        'is_night': is_night,
        'log_unitprice': log_unitprice,
        'log_quantity': log_quantity,
        'log_total_price': log_total_price
    }
    return pd.DataFrame([data])

# Main app logic
st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    input_df = user_input_features()
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Make predictions
if st.button("Predict"):
    # Ensure all required features are present and in correct order
    required_features = ["description", "quantity", "unitprice", "year", "month", "day", "hour", "dayofweek", "total_price", "weekday", "country_encoded", "is_weekend", "is_night", "log_unitprice", "log_quantity", "log_total_price"]
    
    # Reorder columns to match training data
    input_df = input_df[required_features]
    
    predictions = model.predict(input_df)
    predicted_labels = encoder.inverse_transform(predictions)
    input_df['Predicted Segment'] = predicted_labels
    st.success("Prediction Completed!")
    st.write(input_df)

    # Display feature importance
    st.subheader("Feature Importance")
    import matplotlib.pyplot as plt
    import seaborn as sns

    importance_df = pd.DataFrame({
        'Feature': model.feature_name_,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    st.pyplot(plt)
