import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go
import os

try:
    from st_aggrid import AgGrid
except ImportError:
    AgGrid = None

st.set_page_config("Interactive Stock Predictor", layout="wide")
st.title("📊 Interactive Stock Price Predictor with LSTM")

# Sidebar user inputs
with st.sidebar:
    st.header("Your Inputs")
    ticker = st.text_input("Ticker symbol (e.g. AAPL)", value="AAPL", max_chars=8)
    start_date = st.date_input("Start date", value=date(2015,1,1))
    end_date = st.date_input("End date", value=date.today())
    run = st.button("Fetch & Predict")

if (not run) and "df" not in st.session_state:
    st.info("Enter inputs and press 'Fetch & Predict' to begin.")
    st.stop()

if run:
    with st.spinner(f"Fetching data for {ticker}..."):
        df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found. Please choose a valid ticker and dates.")
        st.stop()
    st.session_state["df"] = df

df = st.session_state["df"]
st.success(f"Loaded {len(df)} rows of data from {df.index.min().date()} to {df.index.max().date()}")

# Data table
st.subheader("Latest Price Data")
if AgGrid:
    AgGrid(df.tail(100))
else:
    st.dataframe(df.tail(100))

# Compute moving averages conditionally with no errors
close = df["Close"]
ma100 = close.rolling(100).mean() if len(close) >= 100 else None
ma200 = close.rolling(200).mean() if len(close) >= 200 else None

# Interactive Plotly chart of closing price + MAs
st.subheader("Interactive Closing Price & Moving Averages Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=close.index, y=close, name='Close', mode='lines', line=dict(color='royalblue')))

if ma100 is not None and ma100.size > 0 and np.any(ma100.notna().values):
    fig.add_trace(go.Scatter(x=ma100.index, y=ma100, name='100-day MA', mode='lines', line=dict(color='firebrick')))
else:
    if len(close) < 100:
        st.info("100-day MA unavailable (not enough data).")

if ma200 is not None and ma200.size > 0 and np.any(ma200.notna().values):
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name='200-day MA', mode='lines', line=dict(color='forestgreen')))
else:
    if len(close) < 200:
        st.info("200-day MA unavailable (not enough data).")

fig.update_layout(
    xaxis_title='Date', yaxis_title='Price ($)',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# LSTM Model Training + Prediction if enough data
window = 100
if len(close) < window + 1:
    st.warning(f"Need at least {window + 1} data points for LSTM prediction. Increase date range or choose another ticker.")
    st.stop()

scaler = MinMaxScaler()
train_size = int(len(close) * 0.7)
train_data = close[:train_size].values.reshape(-1, 1)
test_data = close[train_size - window:].values.reshape(-1, 1)

scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

def make_dataset(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = make_dataset(train_scaled, window)
X_test, y_test = make_dataset(test_scaled, window)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

st.write(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

model_path = f"{ticker}_lstm.keras"

if os.path.isfile(model_path):
    model = load_model(model_path)
    st.info("Loaded existing LSTM model.")
else:
    with st.spinner("Training LSTM model — please wait..."):
        model = Sequential()
        model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(window, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(60, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=0)
        model.save(model_path)
    st.success("Model training complete and saved.")

# Prediction and inverse scaling
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Prediction plot
st.subheader("Predicted vs Actual Closing Prices")
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(y=y_test_inv.flatten(), mode='lines', name='Actual', line=dict(color='royalblue')))
fig_pred.add_trace(go.Scatter(y=y_pred_inv.flatten(), mode='lines', name='Predicted', line=dict(color='orangered')))
fig_pred.update_layout(
    xaxis_title="Time steps",
    yaxis_title="Price ($)",
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_pred, use_container_width=True)

st.caption("This is a demonstration app for educational purposes, not financial advice.")
