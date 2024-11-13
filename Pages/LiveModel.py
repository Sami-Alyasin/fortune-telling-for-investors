import joblib
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sklearn as sk
import lightgbm as lgb
from scipy import stats
import io
import requests
import onnxruntime as rt

st.set_page_config(layout="centered",initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center;'>Fortune Telling for Investors</h1>", unsafe_allow_html=True)

st.divider()
left, middle, right = st.columns(3,gap='large')  
with left:
    st.page_link("Pages/Home.py",label = "About", icon = "📝")
    
with middle:
    st.page_link("Pages/Project.py",label = "Project walkthorugh", icon = "📚")  
    
with right:
    st.page_link("Pages/LiveModel.py",label = "Give the model a try!", icon = "🔮")

# Load the saved RF model
# # URL to the raw model file in GitHub
model_url = "https://raw.githubusercontent.com/Sami-Alyasin/fortune-telling-for-investors/main/model/rf_model.onnx"

try:
    # Download the model file from GitHub
    response = requests.get(model_url)
    response.raise_for_status()
    model_bytes = response.content

    # Load the model from the downloaded bytes
    session = rt.InferenceSession(model_bytes)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error: {e}")

sector_data = pd.read_csv('https://raw.githubusercontent.com/Sami-Alyasin/fortune-telling-for-investors/main/data/sector_data.csv')


# Your existing data preparation
data = yf.download("AAPL MSFT GOOGL META AMZN JPM BAC WFC GS MS JNJ PFE MRK ABT BMY PG KO PEP NKE UL XOM CVX SLB COP BP BA CAT MMM GE HON", 
                   period="5y", group_by='ticker')

df_temp = data.stack(level=0).reset_index()
df = pd.merge(df_temp, sector_data, how='left', on='Ticker')
df = df.sort_values(by=['Ticker', 'Date'])
tickers = df['Ticker'].unique()

selected_ticker = st.selectbox("Select a stock ticker:", tickers)

# Calculating features
df['Daily_Return'] = df['Adj Close'].pct_change()
df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()
df['Volatility_15'] = df['Daily_Return'].rolling(window=15).std()
df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
df['Z_Score_ACP'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: stats.zscore(x))
df['Z_Score_Volume'] = df.groupby('Ticker')['Volume'].transform(lambda x: stats.zscore(x))

def calculate_bollinger_bands(df, window):
    df[f'BB_upper_{window}'] = df[f'RM_{window}'] + (df['Adj Close'].rolling(window=window).std() * 2)
    df[f'BB_lower_{window}'] = df[f'RM_{window}'] - (df['Adj Close'].rolling(window=window).std() * 2)
    return df

for window in [5, 10, 15, 30, 60, 90]:
    df[f'RM_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=window).mean())
    df[f'RSTD_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=window).std())
    df = df.groupby('Ticker', group_keys=False).apply(lambda x: calculate_bollinger_bands(x, window)).reset_index(drop=True)

def calculate_rsi(df, window):
    delta = df.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

for window in [5, 10, 15, 30, 60, 90]:
    df[f'RSI_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: calculate_rsi(x, window))

df.dropna(inplace=True)

# Prediction function
def predict_tomorrow_stock_price(ticker):
    stock_data = df[df['Ticker'] == ticker]
    last_row = stock_data.iloc[-1]

    # Prepare the next day's features
    X_tomorrow = pd.DataFrame({
        'Daily_Return': last_row['Daily_Return'],
        'BB_upper_15': stock_data['Adj Close'].rolling(window=15).mean().iloc[-1] + (2 * stock_data['Adj Close'].rolling(window=15).std().iloc[-1]),
        'BB_upper_10': stock_data['Adj Close'].rolling(window=10).mean().iloc[-1] + (2 * stock_data['Adj Close'].rolling(window=10).std().iloc[-1]),
        'RM_30': stock_data['Adj Close'].rolling(window=30).mean().iloc[-1],
        'BB_upper_5': stock_data['Adj Close'].rolling(window=5).mean().iloc[-1] + (2 * stock_data['Adj Close'].rolling(window=5).std().iloc[-1]),
        'RSTD_5': stock_data['Adj Close'].rolling(window=5).std().iloc[-1],
        'BB_lower_5': stock_data['Adj Close'].rolling(window=5).mean().iloc[-1] - (2 * stock_data['Adj Close'].rolling(window=5).std().iloc[-1]),
        'Z_Score_ACP': last_row['Z_Score_ACP'],
        'RM_5': stock_data['Adj Close'].rolling(window=5).mean().iloc[-1],
        'RSI_5': calculate_rsi(stock_data['Adj Close'], window=5).iloc[-1],
        'RM_15': stock_data['Adj Close'].rolling(window=15).mean().iloc[-1],
        'Volatility_5': last_row['Volatility_5'],
        'RM_60': stock_data['Adj Close'].rolling(window=60).mean().iloc[-1],
        'RSI_10': calculate_rsi(stock_data['Adj Close'], window=10).iloc[-1],
        'BB_lower_10': stock_data['Adj Close'].rolling(window=10).mean().iloc[-1] - (2 * stock_data['Adj Close'].rolling(window=10).std().iloc[-1]),
        'Z_Score_Volume': last_row['Z_Score_Volume'],
        'RSI_15': calculate_rsi(stock_data['Adj Close'], window=15).iloc[-1],
        'RM_10': stock_data['Adj Close'].rolling(window=10).mean().iloc[-1],
        'Volatility_15': last_row['Volatility_15'],
        'RSI_90': calculate_rsi(stock_data['Adj Close'], window=90).iloc[-1],
        'RM_90': stock_data['Adj Close'].rolling(window=90).mean().iloc[-1]
    }, index=[0])

    # Convert the input data to the format required by ONNX
    X_tomorrow_np = X_tomorrow.values.astype(np.float32)

    # Get the input name for ONNX model
    input_name = session.get_inputs()[0].name

    # Run inference
    predictions = session.run(None, {input_name: X_tomorrow_np})

    return float(predictions[0][0])

# Streamlit button for prediction
if st.button('Predict Tomorrow\'s Price'):
    tomorrow_price = predict_tomorrow_stock_price(selected_ticker)
    st.markdown(f"<h2 style='text-align: center;'>🎯 Predicted adjusted close price for {selected_ticker} is: ${tomorrow_price:.2f}</h2>", unsafe_allow_html=True)
    st.write(f"Last known adjusted closing price for {selected_ticker} was ${df[df['Ticker'] == selected_ticker]['Adj Close'].iloc[-1]:.2f}")