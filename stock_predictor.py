# Importing the necessary libraries
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetching stock data using Alpha Vantage API
api_key = '#################'  # Replace with your Alpha Vantage API key

# Initialize the TimeSeries object and fetch daily stock data for AAPL (Apple)
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

# Save the raw data to a CSV file for future reference
data.to_csv('stock_data.csv')

# Load and preprocess the stock data
data = pd.read_csv('stock_data.csv')

# Convert the 'date' column to datetime format and set it as the index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Use the closing prices ('4. close') for prediction
close_prices = data['4. close'].values

# Normalize the closing prices
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

# Function to prepare the dataset for LSTM model input
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Set the time step to 60 days and create the input/output datasets
time_step = 60
X, Y = create_dataset(close_prices_scaled, time_step)

# Reshape the dataset to fit the LSTM model input
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model with the Adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model over 50 epochs with a batch size of 64
model.fit(X, Y, batch_size=64, epochs=50)

# Make predictions using the trained model
predictions = model.predict(X)

# Reverse the normalization to convert predictions back to original stock prices
predictions = scaler.inverse_transform(predictions)

# Compare the actual stock prices with the predicted ones
actual_prices = close_prices[time_step+1:]

# Save the predictions and actual prices to a CSV for comparison
comparison = pd.DataFrame({'Actual': actual_prices, 'Predicted': predictions.flatten()})
comparison.to_csv('predictions.csv')

# Visualize the results: actual vs predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(comparison['Actual'], label='Actual Prices')
plt.plot(comparison['Predicted'], label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
