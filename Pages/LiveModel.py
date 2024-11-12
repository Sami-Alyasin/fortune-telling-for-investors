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

st.set_page_config(layout="centered",initial_sidebar_state="expanded")

st.warning('This page temporarily under construction. Working on deploying the Random Forest model.')

# # Load the saved LightGBM model
# # URL to the raw model file in GitHub
# url = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/model/lgb_model.pkl"

# # Send a GET request to download the file
# response = requests.get(url)

# # Ensure the request was successful
# if response.status_code == 200:
#     # Use BytesIO to handle the file as a binary stream and load the model
#     model_file = io.BytesIO(response.content)
#     lgb_model = joblib.load(model_file)
#     print("Model loaded successfully.")
# else:
#     print(f"Failed to download the model. Status code: {response.status_code}")

# # Load sector data
# sector_data = pd.read_csv('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/sector_data.csv')

# # Streamlit layout
# st.markdown("<h1 style='text-align: center;'>Fortune Telling for Investors</h1>", unsafe_allow_html=True)

# st.divider()
# left, middle, right = st.columns(3,gap='large')  
# with left:
#     st.page_link("Pages/Home.py",label = "About", icon = "📝")
    
# with middle:
#     st.page_link("Pages/Project.py",label = "Project walkthorugh", icon = "📚")  
    
# with right:
#     st.page_link("Pages/LiveModel.py",label = "Give the model a try!", icon = "🔮")
    
    
# # download the data for the 30 companies
# data = yf.download("AAPL MSFT GOOGL META AMZN JPM BAC WFC GS MS JNJ PFE MRK ABT BMY PG KO PEP NKE UL XOM CVX SLB COP BP BA CAT MMM GE HON", 
#                    period="5y",
#                    group_by='ticker')

# df_temp = data.stack(level=0).reset_index()

# df = pd.merge(df_temp, sector_data, how='left', on='Ticker')

# df = df.sort_values(by=['Ticker', 'Date'])

# # List of stocks for the user to select from
# tickers = df['Ticker'].unique()

# # User selects a stock ticker
# selected_ticker = st.selectbox("Select a stock ticker:", tickers)

# # Use the Pandas pct_change() function to calculate the change of the Adjusted closing price between each day 
# df['Daily_Return'] = df['Adj Close'].pct_change()

# # Calculate volatility (standard deviation of daily returns)
# df['Volatility'] = df['Daily_Return'].rolling(window=30).std()

# # Z-scores - for outlier detection
# df['Z_Score_ACP'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: stats.zscore(x))
# df['Z_Score_Volume'] = df.groupby('Ticker')['Volume'].transform(lambda x: stats.zscore(x))

# # Define a function to calculate Bollinger Bands for a given window to help guage the volatility of a stock
# def calculate_bollinger_bands(df, window):
#     df[f'BB_upper_{window}'] = df[f'RM_{window}'] + (df['Adj Close'].rolling(window=window).std() * 2)
#     df[f'BB_lower_{window}'] = df[f'RM_{window}'] - (df['Adj Close'].rolling(window=window).std() * 2)
#     return df

# for window in [30, 60, 90]:
#     # Rolling Mean 
#     df[f'RM_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=window).mean())
#     # Rolling Standard Deviation
#     df[f'RSTD_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=window).std())
#     # Bollinger Bands 
#     df = df.groupby('Ticker', group_keys=False).apply(lambda x: calculate_bollinger_bands(x, window)).reset_index(drop=True)

# # RSI (Relative Strength Index) - momentum indicator that measures the speed and change of price movements
# def calculate_rsi(df, window):
#     # Calculate the difference in price from the previous step
#     delta = df.diff(1)
    
#     # Make two series: one for gains and one for losses
#     gain = delta.clip(lower=0)
#     loss = -delta.clip(upper=0)
    
#     # Calculate the rolling average of gains and losses
#     avg_gain = gain.rolling(window=window, min_periods=1).mean()
#     avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
#     # Calculate the Relative Strength (RS)
#     rs = avg_gain / avg_loss
    
#     # Calculate the RSI
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

# # Calculate RSI for different window sizes and add them to the dataframe
# for window in [30, 60, 90]:
#     df[f'RSI_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: calculate_rsi(x, window))
    
# df.dropna(inplace=True)

# # df.to_csv('stock_data.csv', index=False)

# def predict_tomorrow_stock_price(ticker):
    
#     # Filter for the desired stock
#     stock_data = df[df['Ticker'] == ticker]
    
#     # Get the latest feature values
#     last_row = stock_data.iloc[-1]
    
#     # Prepare the next day's features
#     X_tomorrow = pd.DataFrame({
#     'Daily_Return': last_row['Daily_Return'],  # You can leave this as is or forecast it
    
#     'Volatility_5': stock_data['Daily_Return'].rolling(window=5).std().iloc[-1],
#     'Volatility_15': stock_data['Daily_Return'].rolling(window=15).std().iloc[-1],
#     'Volatility_30': stock_data['Daily_Return'].rolling(window=30).std().iloc[-1],
    
#     'Z_Score_ACP': last_row['Z_Score_ACP'], 
    
#     'Z_Score_Volume': last_row['Z_Score_Volume'], 
    
#     'RM_30': stock_data['Adj Close'].rolling(window=30).mean().iloc[-1],
#     'RM_60': stock_data['Adj Close'].rolling(window=60).mean().iloc[-1],
#     'RM_90': stock_data['Adj Close'].rolling(window=90).mean().iloc[-1],
    
#     'RSTD_30': stock_data['Adj Close'].rolling(window=30).std().iloc[-1],
#     'RSTD_60': stock_data['Adj Close'].rolling(window=60).std().iloc[-1],
#     'RSTD_90': stock_data['Adj Close'].rolling(window=90).std().iloc[-1],
    
#     'BB_upper_30': stock_data['Adj Close'].rolling(window=30).mean().iloc[-1] + (2 * stock_data['Adj Close'].rolling(window=30).std().iloc[-1]),
#     'BB_lower_30': stock_data['Adj Close'].rolling(window=30).mean().iloc[-1] - (2 * stock_data['Adj Close'].rolling(window=30).std().iloc[-1]),
#     'BB_upper_60': stock_data['Adj Close'].rolling(window=60).mean().iloc[-1] + (2 * stock_data['Adj Close'].rolling(window=60).std().iloc[-1]),
#     'BB_lower_60': stock_data['Adj Close'].rolling(window=60).mean().iloc[-1] - (2 * stock_data['Adj Close'].rolling(window=60).std().iloc[-1]),
#     'BB_upper_90': stock_data['Adj Close'].rolling(window=90).mean().iloc[-1] + (2 * stock_data['Adj Close'].rolling(window=90).std().iloc[-1]),
#     'BB_lower_90': stock_data['Adj Close'].rolling(window=90).mean().iloc[-1] - (2 * stock_data['Adj Close'].rolling(window=90).std().iloc[-1]), 
    
#     'RSI_30': calculate_rsi(stock_data['Adj Close'], window=30).iloc[-1],
#     'RSI_60': calculate_rsi(stock_data['Adj Close'], window=60).iloc[-1],
#     'RSI_90': calculate_rsi(stock_data['Adj Close'], window=90).iloc[-1]
#     }, index=[0])   

#     # Predict tomorrow's price
#     tomorrow_prediction = lgb_model.predict(X_tomorrow)
#     return tomorrow_prediction[0]


# if st.button('Predict Tomorrow\'s Price'):
#     tomorrow_price = predict_tomorrow_stock_price(selected_ticker)
#     st.markdown(f"<h2 style='text-align: center;'>🎯 Predicted adjusted close price for {selected_ticker} is: ${tomorrow_price:.2f}</h2>", unsafe_allow_html=True)
#     # print the last known adjusted closing price before the prediction
#     st.write(f"Last known adjusted closing price for {selected_ticker} was ${df[df['Ticker'] == selected_ticker]['Adj Close'].iloc[-1]:.2f}")