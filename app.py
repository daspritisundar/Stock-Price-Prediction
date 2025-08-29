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

st.set_page_config("Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor with LSTM")

# Sidebar with instructions and ticker input
with st.sidebar:
    st.header("Instructions & Input")
    st.markdown("""
    1. Enter a stock ticker symbol (e.g. AAPL, MSFT, TSLA).  
    2. Click **Fetch Data** to analyze and predict stock prices.  
    3. App trains a model if no pre-trained model exists for ticker.  
    4. Visualize and download stock data and predictions below.
    """)
    ticker = st.text_input("Ticker symbol", 'AAPL', max_chars=8)
    fetch = st.button("Fetch Data")

if (not fetch) and "df" not in st.session_state:
    st.info("Please enter a ticker symbol and click 'Fetch Data' to begin.")
    st.stop()

# Data fetch and caching
if fetch:
    try:
        df = yf.download(ticker, start='2015-01-01', end=date.today())
        if df.empty:
            st.error(f"No data found for ticker '{ticker}'. Try a different symbol.")
            st.stop()
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Data loading error: {e}")
        st.stop()

df = st.session_state['df']
st.success(f"{ticker.upper()} data loaded: {df.index.min().date()} - {df.index.max().date()} ({len(df)} rows)")

# Show all columns in a data table
st.subheader("Latest Data (All columns)")
if AgGrid:
    AgGrid(df.tail(100))
else:
    st.dataframe(df.tail(100))

# Calculate moving averages safely
close = df['Close']
ma100 = close.rolling(100).mean() if len(close) >= 100 else None
ma200 = close.rolling(200).mean() if len(close) >= 200 else None

# Plot closing price and MAs
st.subheader("Closing Price and Moving Averages (Interactive Chart)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=close.index, y=close, mode='lines', name='Close', line=dict(color='royalblue')))

if ma100 is not None and ma100.size > 0 and ma100.notna().any():
    fig.add_trace(go.Scatter(x=ma100.index, y=ma100, mode='lines', name='100-Day MA', line=dict(color='firebrick')))
else:
    if len(close) < 100:
        st.info("Not enough data for 100-Day Moving Average")

if ma200 is not None and ma200.size > 0 and ma200.notna().any():
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, mode='lines', name='200-Day MA', line=dict(color='forestgreen')))
else:
    if len(close) < 200:
        st.info("Not enough data for 200-Day Moving Average")

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price ($)",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h")
)
st.plotly_chart(fig, use_container_width=True)

# Warn if data too small for meaningful prediction
if len(close) < 120:
    st.warning("Less than 120 rows of data available. Model training and predictions may be inaccurate or skipped.")

# Prepare LSTM inputs only if enough data
window = 100
if len(close) < window + 1:
    st.info(f"Need at least {window+1} data points for LSTM training. Please choose another ticker or larger date range.")
    st.stop()

# Prepare train/test sets
scaler = MinMaxScaler()
train_len = int(len(close)*0.7)
train_data = close[:train_len].values.reshape(-1,1)
test_data = close[train_len - window:].values.reshape(-1,1)

scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

def create_dataset(dataset, window):
    x,y = [], []
    for i in range(window, len(dataset)):
        x.append(dataset[i-window:i, 0])
        y.append(dataset[i,0])
    return np.array(x), np.array(y)

X_train, y_train = create_dataset(train_scaled, window)
X_test, y_test = create_dataset(test_scaled, window)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))

st.write(f"Training samples: {X_train.shape[0]}    Testing samples: {X_test.shape[0]}")

# Load or train model
model_path = f"{ticker}_lstm.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.info("Loaded pre-trained LSTM model.")
else:
    with st.spinner("Training LSTM model. Please wait..."):
        model = Sequential()
        model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(window,1)))
        model.add(Dropout(0.2))
        model.add(LSTM(60, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        model.save(model_path)
    st.success("Model trained and saved!")

# Predict
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

# Plot predictions
st.subheader("Predicted vs Actual Closing Price")
pred_fig = go.Figure()
pred_fig.add_trace(go.Scatter(y=y_test_inv.flatten(), mode='lines', name='Actual', line=dict(color='royalblue')))
pred_fig.add_trace(go.Scatter(y=y_pred_inv.flatten(), mode='lines', name='Predicted', line=dict(color='orangered')))
pred_fig.update_layout(xaxis_title="Time", yaxis_title="Price ($)", margin=dict(t=20,r=20,b=20,l=10))
st.plotly_chart(pred_fig, use_container_width=True)

st.caption("Note: This model is for educational purposes only and should not be used for investment decisions.")
