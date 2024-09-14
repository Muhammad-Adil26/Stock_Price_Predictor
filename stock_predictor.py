# Import necessary libraries
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Fetch Stock Data using Alpha Vantage API
# Replace 'YOUR_API_KEY' with the actual API key you got from Alpha Vantage
api_key = 'NNZNFTTXIPQ0CMO5'

# Initialize the TimeSeries object with the API key
ts = TimeSeries(key=api_key, output_format='pandas')

# Get daily stock data for a specific symbol (e.g., 'AAPL' for Apple)
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

# Save the raw data to a CSV file for future reference
data.to_csv('stock_data.csv')

# Step 2: Preprocess the data
# Load the stock data
data = pd.read_csv('stock_data.csv')

# Convert the 'date' column to datetime and set it as the index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Use only the '4. close' column for prediction (the closing price)
close_prices = data['4. close'].values

# Normalize the data (scale it between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

# Create a function to prepare the data for LSTM
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Set the time_step (e.g., 60 days of data to predict the next day)
time_step = 60
X, Y = create_dataset(close_prices_scaled, time_step)

# Reshape the data for the LSTM model
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 3: Build the LSTM model using TensorFlow and Keras
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the model
model.fit(X, Y, batch_size=64, epochs=50)

# Step 5: Make Predictions
# Make predictions with the trained model
predictions = model.predict(X)

# Reverse the normalization to get actual stock prices
predictions = scaler.inverse_transform(predictions)

# Compare the actual stock prices with the predicted ones
actual_prices = close_prices[time_step+1:]

# Save predictions and actual prices to CSV for comparison
comparison = pd.DataFrame({'Actual': actual_prices, 'Predicted': predictions.flatten()})
comparison.to_csv('predictions.csv')

# Step 6: Visualize the Results
# Plot the actual vs predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(comparison['Actual'], label='Actual Prices')
plt.plot(comparison['Predicted'], label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()