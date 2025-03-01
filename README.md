﻿# Agricultural-Price-Intelligence
# Price Forecasting Project  

## Overview  
This project predicts the average prices of fruits based on datasets divided by variety and grade. Each dataset corresponds to a specific fruit variety and grade (e.g., Grade A or Grade B). Multiple machine learning and statistical models are employed to ensure robust and accurate forecasting.  

## Features  
- Tailored price predictions for each fruit variety and grade.  
- Incorporates availability (`Mask` column) and time-based features.  
- Supports both batch forecasting and real-time price prediction through a web interface.  

---

## Models Used  

### 1. Tree-Based Models  
- **Random Forest**: Captures complex interactions and handles missing data effectively.  
- **XGBoost**: Provides high predictive performance for datasets with high variance.  

### 2. Seasonal Models  
- **SARIMA**: Captures seasonality and trends in price data.  
- **Prophet**: Flexible handling of irregular seasonal patterns and missing data.  

### 3. Deep Learning Models  
- **LSTM**: Learns sequential dependencies in price data.  
- **Transformer**: State-of-the-art for time-series forecasting with long-term dependencies.  

---

## Project Structure  

### **Data**  
- **`data/raw/`**: Contains raw datasets.  
- **`data/processed/`**: Preprocessed datasets ready for analysis.  
- **`data/samples/`**: Sample data for quick experimentation.  
- **`weekly_data.csv`**: Aggregated weekly dataset for price analysis.  

### **Models**  
- **`models/`**: Stores trained model files and configurations.  

### **Notebooks**  
- **`eda.ipynb`**: Exploratory Data Analysis (EDA) to uncover patterns and trends.  
- **`model_experiments.ipynb`**: Experiments with various machine learning and deep learning models.  

### **Source Code**  
- **`src/`**:  
  - **`data_pipeline.py`**: Data preprocessing and feature engineering pipeline.  
  - **`forecasting.py`**: Main script for forecasting price data.  
  - **`model_training.py`**: Code for training and saving models.  
  - **`monitor.py`**: Tracks model performance and logs results.  
  - **`real_time_input.py`**: Handles real-time data input for forecasting.  
  - **`utils.py`**: Utility functions for data manipulation and evaluation.  

### **Testing**  
- **`tests/`**:  
  - **`test_forecasting.py`**: Validates model prediction logic.  
  - **`test_models.py`**: Tests training and evaluation modules.  
  - **`test_pipeline.py`**: Ensures data preprocessing accuracy.  

### **Web Application**  
- **`web/`**:  
  - **`app.py`**: Flask-based application for deploying the forecasting models.  
  - **`config.py`**: Configuration settings for the web application.  
  - **`routes.py`**: API routes and endpoints for real-time prediction.  
  - **`static/`**: Static assets for the web application (e.g., CSS/JavaScript).  

---

## How to Use  

### Installation  
1. Clone the repository:  
   ```bash  
   git clone <repository-url>  
   cd PRICE_FORECASTING_PROJECT  
