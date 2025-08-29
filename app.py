import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os

st.set_page_config("Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor with LSTM")

# Sidebar
with st.sidebar:
    st.header("Instructions & Input")
    st.markdown("""
    1. Enter a stock ticker (e.g. AAPL, MSFT, TSLA).  
    2. Click 'Fetch Data' to analyze and predict stock prices.  
    3. Model trains if no pre-trained model exists for the ticker.  
    4. View prices, moving averages, and interactive predictions below.
    """)
    ticker = st.text_input("Ticker symbol", 'AAPL', max_chars=8)
    fetch = st.button("Fetch Data")

if (not fetch) and "df" not in st.session_state:
    st.info("Enter a ticker and click 'Fetch Data' to start.")
    st.stop()

if fetch:
    try:
        df = yf.download(ticker, start='2015-01-01', end=date.today())
        if df.empty:
            st.error(f"No data for ticker '{ticker}'.")
            st.stop()
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

df = st.session_state["df"]
st.success(f"{ticker.upper()} data loaded: {df.index.min().date()} - {df.index.max().date()} ({len(df)} rows)")

# Show all columns as table
st.subheader("Latest Data (All columns)")
st.dataframe(df.tail(100))

# Prepare Closing Price and Moving Averages (Matplotlib - simple)
close = df["Close"]
ma100 = close.rolling(window=100).mean() if len(close) >= 100 else None
ma200 = close.rolling(window=200).mean() if len(close) >= 200 else None

st.subheader("Closing Price and Moving Averages (Simple)")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, close, label="Close Price", color="blue")

if ma100 is not None:
    ax.plot(df.index, ma100, label="100-day MA", color="red")
else:
    st.info("Not enough data for 100-day moving average.")

if ma200 is not None:
    ax.plot(df.index, ma200, label="200-day MA", color="green")
else:
    st.info("Not enough data for 200-day moving average.")

ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Prepare data for LSTM
window = 100
if len(close) < window + 1:
    st.info(f"Need at least {window+1} data points for LSTM model training. Please select a different ticker or longer time range.")
    st.stop()

scaler = MinMaxScaler()
train_len = int(len(close) * 0.7)
train_data = close[:train_len].values.reshape(-1, 1)
test_data = close[train_len - window :].values.reshape(-1, 1)

scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

def create_dataset(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_scaled, window)
X_test, y_test = create_dataset(test_scaled, window)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

st.write(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

# Load or train model
model_path = f"{ticker}_lstm.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.info("Loaded existing LSTM model.")
else:
    with st.spinner("Training LSTM model..."):
        model = Sequential()
        model.add(LSTM(60, activation="relu", return_sequences=True, input_shape=(window, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(60, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        model.save(model_path)
    st.success("Model trained and saved.")

# Predict and inverse scaling
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Interactive prediction plot using Plotly
st.subheader("Predicted vs Actual Closing Prices (Interactive)")

fig_pred = go.Figure()
fig_pred.add_trace(
    go.Scatter(
        y=y_test_inv.flatten(), mode="lines", name="Actual Price", line=dict(color="royalblue")
    )
)
fig_pred.add_trace(
    go.Scatter(
        y=y_pred_inv.flatten(), mode="lines", name="Predicted Price", line=dict(color="orangered")
    )
)
fig_pred.update_layout(
    xaxis_title="Time steps", yaxis_title="Price ($)", margin=dict(t=20, r=20, b=20, l=20)
)
st.plotly_chart(fig_pred, use_container_width=True)

st.caption("Model and app for educational purposes only. Not financial advice.")
