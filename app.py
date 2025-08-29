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

st.set_page_config("Stock Price Forecast", layout="wide")
st.title("📈 Stock Price Forecast with LSTM")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Enter a valid stock ticker symbol (e.g., **AAPL**, **MSFT**, **TSLA**).
    2. Press **Fetch Data** to load, view, and analyze historical prices.
    3. View the forecast results below.
    """)
    ticker = st.text_input('Ticker symbol', value='AAPL', max_chars=8)
    run_button = st.button("Fetch Data")

if (not run_button) and "df" not in st.session_state:
    st.info("Enter a ticker and click 'Fetch Data' to begin.")
    st.stop()
if run_button:
    try:
        df = yf.download(ticker, start='2015-01-01', end=date.today())
        if df.empty:
            st.error("Ticker not found or no data. Try another symbol.")
            st.stop()
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df = st.session_state['df']
st.success(f"{ticker.upper()} data loaded: {df.index.min().date()} - {df.index.max().date()}  ({len(df)} rows).")
st.download_button("Download CSV", data=df.to_csv().encode(), file_name=f"{ticker}.csv")

# Visualization section
st.subheader("Closing Price and Moving Averages (Interactive)")
fig = go.Figure([
    go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color="royalblue")),
    go.Scatter(x=df.index, y=df['Close'].rolling(100).mean(), name='100-Day MA', line=dict(color="firebrick")),
    go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), name='200-Day MA', line=dict(color="forestgreen"))
])
fig.update_layout(margin=dict(t=30,r=30,b=30,l=10), legend=dict(orientation="h",yanchor="bottom"), xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(fig, use_container_width=True)

# Prepare data for LSTM
close_prices = df[['Close']]
split = int(len(close_prices) * 0.7)
train = close_prices.iloc[:split]
test = close_prices.iloc[split:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)

window = 100
def make_sequences(data, window=100):
    xs, ys = [], []
    for i in range(window, len(data)):
        xs.append(data[i-window:i])
        ys.append(data[i,0])
    return np.array(xs), np.array(ys)

X_train, y_train = make_sequences(train_scaled, window)

model_path = f"{ticker}_lstm.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.info("Loaded pre-trained LSTM model.")
else:
    with st.spinner("Training new LSTM model... Please wait."):
        model = Sequential()
        model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(60, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        model.save(model_path)
    st.success("Model trained and saved.")

# Testing
total_data = pd.concat([train.tail(window), test])
inputs = scaler.transform(total_data)
X_test, y_test = make_sequences(inputs, window)

# Prediction
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(np.concatenate([y_pred, np.zeros((y_pred.shape[0], 0))], axis=1))[:,0]
y_test_inv = scaler.inverse_transform(np.concatenate([y_test.reshape(-1,1), np.zeros((y_test.shape[0], 0))], axis=1))[:,0]

# Plot predictions
st.subheader("Predicted vs Actual Closing Price")
line_fig = go.Figure([
    go.Scatter(y=y_test_inv, mode="lines", name="Actual", line=dict(color="royalblue")),
    go.Scatter(y=y_pred_inv, mode="lines", name="Predicted", line=dict(color="orangered"))
])
line_fig.update_layout(xaxis_title="Time", yaxis_title="Price ($)", margin=dict(t=20,r=20,b=20,l=10))
st.plotly_chart(line_fig, use_container_width=True)

st.caption("Tip: Retrain model by deleting the .keras file. This is a research demo—do not use for investing decisions.")

