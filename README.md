# Shopper's Spectrum Classification App

A machine learning web application that predicts customer segments based on online purchase behavior using LightGBM classification.

## Overview

This application analyzes customer purchase patterns from online retail data to classify customers into different segments. It uses a trained LightGBM model to predict customer segments based on various features including purchase quantities, prices, timing, and geographic information.

## Features

- **Customer Segment Prediction**: Predict customer segments based on purchase behavior
- **Interactive Web Interface**: User-friendly Streamlit interface for data input
- **Feature Importance Analysis**: Visualize which features drive predictions
- **Multiple Input Methods**: Manual input or CSV file upload
- **Real-time Predictions**: Instant results with detailed analysis

## Model Features

The model uses 16 features for prediction:
- **Product Information**: Quantity, Unit Price, Total Price
- **Temporal Features**: Year, Month, Day, Hour, Day of Week, Weekday
- **Behavioral Indicators**: Is Weekend, Is Night
- **Geographic Data**: Country Encoded
- **Derived Features**: Log-transformed quantities and prices

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Shopper's-Spectrum
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Choose input method:
   - **Manual Input**: Fill in the form with your data
   - **Upload CSV**: Upload a CSV file with the required features

4. Click "Predict" to get customer segment predictions

## Input Fields

### Manual Input Mode
- **Description**: Product description (for reference)
- **Quantity**: Number of items purchased
- **Unit Price**: Price per unit
- **Year/Month/Day**: Purchase date
- **Hour**: Purchase time (0-23)
- **Day of Week**: Day of week (0=Monday, 6=Sunday)
- **Weekday**: Binary indicator (1=Weekday, 0=Weekend)
- **Is Weekend**: Binary indicator (1=Weekend, 0=Weekday)
- **Is Night**: Binary indicator (1=Night, 0=Day)
- **Country Encoded**: Encoded country value (0-3)

### CSV Upload Mode
Upload a CSV file with the following columns:
- description, quantity, unitprice, year, month, day, hour, dayofweek, total_price, weekday, country_encoded, is_weekend, is_night, log_unitprice, log_quantity, log_total_price

## Model Information

- **Algorithm**: LightGBM Classifier
- **Training Data**: Online retail dataset
- **Features**: 16 engineered features
- **Output**: Customer segment predictions

## File Structure

```
Shopper's Spectrum/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── lightgbm_final_model.pkl # Trained LightGBM model
├── label_encoder.pkl        # Label encoder for predictions
├── feature_columns.json     # Feature column names
├── scaler.pkl              # Data scaler
├── random_forest_model.pkl  # Alternative model
├── online_retail.csv       # Training dataset
├── shopper_spectrum_model_training.ipynb # Google Colab notebook with model training
└── README.md               # This file
```

## Model Training

The models were trained using Google Colab. The training notebook (`shopper_spectrum_model_training.ipynb`) includes:

- **Data Preprocessing**: Cleaning and feature engineering
- **Exploratory Data Analysis**: Understanding the dataset
- **Feature Engineering**: Creating temporal and behavioral features
- **Model Training**: Training LightGBM and Random Forest models
- **Model Evaluation**: Performance metrics and validation
- **Model Export**: Saving trained models for deployment

To retrain the models:
1. Upload the notebook to Google Colab
2. Upload the `online_retail.csv` dataset
3. Run all cells to retrain the models
4. Download the trained models and update the app

## Dependencies

- streamlit >= 1.28.0
- pandas >= 1.5.0
- joblib >= 1.3.0
- numpy >= 1.24.0
- lightgbm >= 4.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0

## Deployment

### Local Deployment
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Access at: `http://localhost:8501`

### Cloud Deployment
The app can be deployed on platforms like:
- Streamlit Cloud
- Heroku
- Google Cloud Platform
- AWS

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please open an issue in the repository. 