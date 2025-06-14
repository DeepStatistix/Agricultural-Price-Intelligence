import os
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    if len(data) <= seq_length:
        print("Error: Data length must be greater than the sequence length.")
        return np.array([]), np.array([])

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])  # Target value

    X, y = np.array(X), np.array(y)
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y

# Function to find the best sequence length
def find_best_seq_length(data, max_seq_length):
    best_seq_length = 0
    best_mse = float('inf')

    for seq_length in range(1, max_seq_length + 1):
        X, y = create_sequences(data[['Avg Price (per kg)']].values, seq_length)
        if len(X) == 0:
            continue

        # Define and train a temporary model
        temp_model = Sequential([
            LSTM(100, activation='relu', input_shape=(seq_length, 1)),
            Dense(1)
        ])
        temp_model.compile(optimizer='adam', loss='mse')
        temp_model.fit(X.reshape((X.shape[0], X.shape[1], 1)), y, epochs=50, batch_size=16, verbose=0)

        # Evaluate the model
        predictions = temp_model.predict(X.reshape((X.shape[0], X.shape[1], 1)))
        # Check for NaN in predictions or in the computed mse
        if np.isnan(predictions).any():
            logging.warning(f"Sequence length {seq_length}: predictions contain NaN, skipping this length.")
            continue

        mse = mean_squared_error(y, predictions)
        if np.isnan(mse):
            logging.warning(f"Sequence length {seq_length}: computed MSE is NaN, skipping this length.")
            continue

        if mse < best_mse:
            best_mse, best_seq_length = mse, seq_length

    return best_seq_length

# Function to train LSTM model
def train_lstm(data, seq_length):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Avg Price (per kg)']].values)

    X, y = create_sequences(data_scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(100, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        Dropout(0.2),  # Dropout layer
        LSTM(50, activation='relu'),
        Dropout(0.2),  # Dropout layer
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    model.fit(X, y, epochs=100, batch_size=16, verbose=0, callbacks=[early_stopping])

    return model, scaler

# Function to forecast future values
def forecast_future(model, scaler, data, seq_length, forecast_days):
    data_scaled = scaler.transform(data[['Avg Price (per kg)']].values)
    input_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)

    predictions = []
    for _ in range(forecast_days):
        next_price = model.predict(input_sequence)[0, 0]
        predictions.append(next_price)
        input_sequence = np.append(input_sequence[:, 1:, :], [[[next_price]]], axis=1)  # Updated line

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions

# Function to evaluate model
def evaluate_model(model, X, y, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    y = scaler.inverse_transform(y.reshape(-1, 1))
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    return mse, mae

def main():
    market = ["Shopian"]  # Markets to process
    varieties = ["Cherry"]  # Varieties
    grades = ["Large","Medium","Small"]  # Grades, if applicable
    forecast_days = 30  # Forecast for 10 days
    max_seq_length = 40  # Maximum sequence length to search for

    results = {}

    for m in market:
        for variety in varieties:
            # First, try to find a no-grade dataset:
            no_grade_path = f"data/raw/processed/{m}/{variety}_dataset.csv"
            if os.path.exists(no_grade_path):
                logging.info(f"Processing {m} {variety} (no grade)...")
                data = pd.read_csv(no_grade_path)
                if 'Avg Price (per kg)' not in data.columns or 'Mask' not in data.columns:
                    logging.error(f"Missing required columns in {no_grade_path}")
                    continue
                if data[['Avg Price (per kg)', 'Mask']].isnull().any().any():
                    logging.error(f"NaN values found in {no_grade_path}.")
                    continue
                data = data[data['Mask'] == 1]
                if data.empty or data['Avg Price (per kg)'].isnull().any():
                    logging.error(f"Filtered data is empty or contains NaN values for {no_grade_path}.")
                    continue

                best_seq_length = find_best_seq_length(data, max_seq_length)
                logging.info(f"Best sequence length for {variety} (no grade): {best_seq_length}")

                model, scaler = train_lstm(data, best_seq_length)

                # Create output directories
                model_forecasts_dir = f"model_forecasts/{m}/{variety}"
                os.makedirs(model_forecasts_dir, exist_ok=True)
                model_path = f"models/{m}/{variety}/lstm_{variety}.h5"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                logging.info(f"Saving model to {model_path}...")
                model.save(model_path)
                logging.info("Model saved successfully.")

                # Make predictions
                predictions = forecast_future(model, scaler, data, best_seq_length, forecast_days)
                std_dev = np.std(predictions)
                lower_bound = predictions - 1.96 * std_dev
                upper_bound = predictions + 1.96 * std_dev

                results[f"{variety}"] = predictions

                # Plot forecasted prices with confidence intervals
                plt.plot(range(forecast_days), predictions, label='Forecasted Prices', color='orange')
                plt.fill_between(range(forecast_days), lower_bound, upper_bound, color='lightgray', alpha=0.5, label='Confidence Interval')
                plt.title(f'Price Forecast for {variety} in {m}')
                plt.xlabel('Days')
                plt.ylabel('Price (per kg)')
                plt.legend()
                plot_path = f"model_forecasts/{m}/{variety}/{variety}_forecast.png"
                plt.savefig(plot_path)
                plt.close()
                logging.info(f"Forecast for {variety}: {predictions}")

                # Save forecasts to CSV
                forecast_df = pd.DataFrame({
                    'Predictions': predictions,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound
                })
                forecast_file_path = f"model_forecasts/{m}/{variety}/{variety}_forecasts.csv"
                forecast_df.to_csv(forecast_file_path, index=False)
                logging.info(f"Forecasts saved to {forecast_file_path}")

            else:
                # Otherwise, iterate over grades
                for grade in grades:
                    logging.info(f"Processing {m} {variety} Grade {grade}...")
                    data_path = f"data/raw/processed/{m}/{variety}_{grade}_dataset.csv"
                    if not os.path.exists(data_path):
                        logging.warning(f"File not found: {data_path}")
                        continue

                    data = pd.read_csv(data_path)
                    if 'Avg Price (per kg)' not in data.columns or 'Mask' not in data.columns:
                        logging.error(f"Missing required columns in {data_path}")
                        continue

                    if data[['Avg Price (per kg)', 'Mask']].isnull().any().any():
                        logging.error(f"NaN values found in {data_path}.")
                        continue

                    data = data[data['Mask'] == 1]
                    if data.empty or data['Avg Price (per kg)'].isnull().any():
                        logging.error(f"Filtered data is empty or contains NaN values for {data_path}.")
                        continue

                    best_seq_length = find_best_seq_length(data, max_seq_length)
                    logging.info(f"Best sequence length for {variety} Grade {grade}: {best_seq_length}")

                    model, scaler = train_lstm(data, best_seq_length)

                    model_forecasts_dir = f"model_forecasts/{m}/{variety}/{grade}"
                    os.makedirs(model_forecasts_dir, exist_ok=True)
                    model_path = f"models/{m}/{variety}/{grade}/lstm_{variety}_grade_{grade}.h5"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    logging.info(f"Saving model to {model_path}...")
                    model.save(model_path)
                    logging.info("Model saved successfully.")

                    predictions = forecast_future(model, scaler, data, best_seq_length, forecast_days)
                    std_dev = np.std(predictions)
                    lower_bound = predictions - 1.96 * std_dev
                    upper_bound = predictions + 1.96 * std_dev

                    results[f"{variety}_grade_{grade}"] = predictions

                    plt.plot(range(forecast_days), predictions, label='Forecasted Prices', color='orange')
                    plt.fill_between(range(forecast_days), lower_bound, upper_bound, color='lightgray', alpha=0.5, label='Confidence Interval')
                    plt.title(f'Price Forecast for {variety} Grade {grade} in {m}')
                    plt.xlabel('Days')
                    plt.ylabel('Price (per kg)')
                    plt.legend()
                    plot_path = f"model_forecasts/{m}/{variety}/{grade}/{variety}_grade_{grade}_forecast.png"
                    plt.savefig(plot_path)
                    plt.close()
                    logging.info(f"Forecast for {variety} Grade {grade}: {predictions}")

                    forecast_df = pd.DataFrame({
                        'Predictions': predictions,
                        'Lower Bound': lower_bound,
                        'Upper Bound': upper_bound
                    })
                    forecast_file_path = f"model_forecasts/{m}/{variety}/{grade}/{variety}_Grade_{grade}_forecasts.csv"
                    forecast_df.to_csv(forecast_file_path, index=False)
                    logging.info(f"Forecasts saved to {forecast_file_path}")

if __name__ == "__main__":
    main()
