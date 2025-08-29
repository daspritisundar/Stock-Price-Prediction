import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os

st.set_page_config(page_title="Stock Prediction Pro", layout="wide")
st.title('📈 Stock Trend Prediction Web App')

# Sidebar for user input
st.sidebar.title("Configuration & Help")
st.sidebar.write("Enter a valid stock ticker (e.g., AAPL, MSFT) to forecast its closing prices using LSTM.")
ticker = st.sidebar.text_input('Ticker', value='AAPL')
start_date = st.sidebar.date_input('Start Date', date(2015,1,1))
end_date = st.sidebar.date_input('End Date', date.today())

# Data download and preview
with st.spinner('Downloading stock data from Yahoo Finance...'):
    df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error("No data found. Please try a different ticker or date range.")
    st.stop()

st.success("Data download successful!")
st.subheader(f'Data for {ticker.upper()} from {start_date} to {end_date} ({len(df)} rows)')
st.dataframe(df.tail())
st.download_button(label="Download CSV", data=df.to_csv().encode(), file_name=f"{ticker}.csv")

# Plotly chart for interactive, beautiful plots
st.subheader("Interactive Price Visualization (Plotly)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(100).mean(), name='100MA', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), name='200MA', line=dict(color='green')))
fig.update_layout(title=f"Closing Price and Moving Averages for {ticker.upper()}",
                  xaxis_title="Date", yaxis_title="Price ($)", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# Data preparation
df_model = df[['Close']].copy()
split = int(len(df_model)*0.7)
data_training = df_model[:split]
data_testing = df_model[split:]

# Scale on training only (ML best practice)
scaler = MinMaxScaler(feature_range=(0,1))
data_training_arr = scaler.fit_transform(data_training)

# Prepare x_train, y_train
x_train, y_train = [], []
window = 100
for i in range(window, len(data_training_arr)):
    x_train.append(data_training_arr[i-window:i])
    y_train.append(data_training_arr[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# LSTM or Load Model with progress indicator
model_path = f"{ticker}_stock_model.keras"
if os.path.exists(model_path):
    with st.spinner('Loading pre-trained LSTM model...'):
        model = load_model(model_path)
    st.toast('Loaded pre-trained model', icon="🧠")
else:
    with st.spinner('Training LSTM model...please wait'):
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=20, verbose=0)
        model.save(model_path)
    st.toast('Model trained and saved!', icon="✅")

# Prepare test set (keep scale from train)
past_100_days = data_training.tail(window)
final_df = pd.concat([past_100_days, data_testing])
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(window, input_data.shape[0]):
    x_test.append(input_data[i-window:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
with st.spinner("Predicting with LSTM model..."):
    y_predicted = model.predict(x_test)

# Inverse scaling
y_predicted = scaler.inverse_transform(np.concatenate([y_predicted, np.zeros((y_predicted.shape[0], 0))], axis=1))[:,0]
y_test_inv = scaler.inverse_transform(np.concatenate([y_test.reshape(-1,1), np.zeros((y_test.shape[0], 0))], axis=1))[:,0]

# Plot result
st.subheader("Predicted vs Actual Closing Price")
fig2 = plt.figure(figsize=(14,6))
plt.plot(y_test_inv, label='Original Price', color='blue')
plt.plot(y_predicted, label='Predicted Price', color='red')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig2)

# Clean up UI/footer
st.info("Built with Streamlit, yfinance, Keras. Adjust model architecture, add more indicators, or experiment further for research or hobby purposes.")

