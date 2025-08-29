import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

st.set_page_config("Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor with LSTM")

# Sidebar for instructions and ticker input
with st.sidebar:
    st.header("Instructions & Input")
    st.markdown("""
    1. Enter a stock ticker symbol (e.g. AAPL, MSFT, TSLA).  
    2. Click **Fetch Data** to download and analyze prices.  
    3. Model trains if no pre-trained model exists.  
    4. View prices, moving averages, and predictions below.
    """)
    ticker = st.text_input("Ticker symbol", 'AAPL', max_chars=8)
    fetch = st.button("Fetch Data")

if (not fetch) and "df" not in st.session_state:
    st.info("Please enter a ticker symbol and click 'Fetch Data' to begin.")
    st.stop()

# Data fetching and caching
if fetch:
    try:
        df = yf.download(ticker, start='2015-01-01', end=date.today())
        if df.empty:
            st.error(f"No data found for ticker '{ticker}'. Try another symbol.")
            st.stop()
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Data loading error: {e}")
        st.stop()

df = st.session_state['df']
st.success(f"{ticker.upper()} data loaded: {df.index.min().date()} - {df.index.max().date()} ({len(df)} rows)")

# Show all columns as table
st.subheader("Latest Data (All columns)")
st.dataframe(df.tail(100))

# Simplified Closing Price and Moving Averages plot with Matplotlib
st.subheader("Closing Price and Moving Averages")

close = df['Close']

ma100 = close.rolling(window=100).mean() if len(df) >= 100 else None
ma200 = close.rolling(window=200).mean() if len(df) >= 200 else None

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, close, label='Closing Price', color='blue')

if ma100 is not None:
    ax.plot(df.index, ma100, label='100-day MA', color='red')
else:
    st.info("Less than 100 data points. 100-day Moving Average not available.")

if ma200 is not None:
    ax.plot(df.index, ma200, label='200-day MA', color='green')
else:
    st.info("Less than 200 data points. 200-day Moving Average not available.")

ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Warn if data too small for model training
window = 100
if len(close) < window + 1:
    st.info(f"Need at least {window+1} data points for LSTM training. Try another ticker or increase date range.")
    st.stop()

# Prepare train/test sets
scaler = MinMaxScaler()
train_len = int(len(close)*0.7)
train_data = close[:train_len].values.reshape(-1,1)
test_data = close[train_len - window:].values.reshape(-1,1)

scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

def create_dataset(data, window):
    x, y = [], []
    for i in range(window, len(data)):
        x.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

X_train, y_train = create_dataset(train_scaled, window)
X_test, y_test = create_dataset(test_scaled, window)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

st.write(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

model_path = f"{ticker}_lstm.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.info("Loaded pre-trained LSTM model.")
else:
    with st.spinner("Training LSTM model. Please wait..."):
        model = Sequential()
        model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(window, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(60, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        model.save(model_path)
    st.success("Model trained and saved!")

# Predict and inverse transform
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predictions
st.subheader("Predicted vs Actual Closing Price")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(y_test_inv, label='Actual Price', color='blue')
ax2.plot(y_pred_inv, label='Predicted Price', color='red')
ax2.set_xlabel("Time steps")
ax2.set_ylabel("Price ($)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.caption("This is an educational demo and not a financial advice.")
