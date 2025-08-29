import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go

try:
    from st_aggrid import AgGrid
except ImportError:
    st.warning("st_aggrid not installed. Tables will be basic. For a better experience, run 'pip install streamlit-aggrid'.")
    AgGrid = None

st.set_page_config(page_title="Stock Price Viewer", layout="wide")
st.title("📈 Stock Price and Moving Average Viewer")

# Sidebar input
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", max_chars=8)
fetch = st.sidebar.button("Fetch Data")

if not fetch and "df" not in st.session_state:
    st.info("Please enter a ticker and press 'Fetch Data'")
    st.stop()

if fetch:
    df = yf.download(ticker, start="2015-01-01", end=pd.Timestamp.today())
    if df.empty:
        st.error(f"No data found for ticker '{ticker}'.")
        st.stop()
    df = df[['Close']].copy()
    df.index.name = "Date"
    st.session_state["df"] = df

df = st.session_state["df"]
st.success(f"Loaded {len(df)} rows of data for {ticker.upper()}.")

# Show table with AgGrid or fallback
st.subheader("Latest Data")
if AgGrid:
    AgGrid(df.tail(100))
else:
    st.dataframe(df.tail(100))

# Calculate moving averages safely
close = df['Close']
ma100 = close.rolling(100).mean() if len(close) >= 100 else None
ma200 = close.rolling(200).mean() if len(close) >= 200 else None

st.subheader("Price & Moving Averages (Interactive Chart)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=close.index, y=close, mode='lines', name='Close', line=dict(color="royalblue")))

if ma100 is not None and ma100.size > 0 and not ma100.isnull().all():
    fig.add_trace(go.Scatter(x=ma100.index, y=ma100, mode='lines', name='100-Day MA', line=dict(color="firebrick")))
else:
    if len(close) < 100:
        st.info("Not enough data for 100-Day Moving Average")

if ma200 is not None and ma200.size > 0 and not ma200.isnull().all():
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, mode='lines', name='200-Day MA', line=dict(color="forestgreen")))
else:
    if len(close) < 200:
        st.info("Not enough data for 200-Day Moving Average")

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price ($)",
    margin=dict(l=10,r=10,t=40,b=10),
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)
