# capturing additional data from the yfinance API
# 1. get the most recent date in the dataframe
# 2. pull the data from the yfinance API starting from that date
# 3. remove the last row of the dataframe (since it will be updated)
# 4. append the new data to the dataframe
# 5. run the feature engineering functions on the new rows only
# 6. to maintain the size of the dataframe, remove as many of the the oldest rows as the new rows added

# import pandas as pd

# yf_df = pd.read_csv('App/data/yfdownload.csv')
# print(yf_df.head())

# if left.button("Plain button", use_container_width=True):
#     left.markdown("[About this project](#about-this-project)", unsafe_allow_html=True)
# if right.button("Material button", icon=":material/mood:", use_container_width=True):
#     right.markdown("[Key Skills Utilized in This Project](#key-skills-utilized-in-this-project)", unsafe_allow_html=True)

# Prepare the data for RandomForestRegressor
# Split data into train and test sets

from sklearn.model_selection import train_test_split
import lightgbm as lgb
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

sector_data = pd.read_csv('/Users/sami/DSP/App/data/sector_data.csv')

# Streamlit layout
# st.markdown("<h1 style='text-align: center;'>The Crystal Stockball: Fortune Telling for Investors</h1>", unsafe_allow_html=True)

# st.divider()
# left, middle, right = st.columns(3,gap='large')  
# with left:
#     st.page_link("Pages/Home.py",label = "About", icon = "📝")
    
# with middle:
#     st.page_link("Pages/Project.py",label = "Project walkthorugh", icon = "📚")  
    
# with right:
#     st.page_link("Pages/LiveModel.py",label = "Give the model a try!", icon = "🔮")
    
    
# download the data for the 30 companies
data = yf.download("AAPL MSFT GOOGL META AMZN JPM BAC WFC GS MS JNJ PFE MRK ABT BMY PG KO PEP NKE UL XOM CVX SLB COP BP BA CAT MMM GE HON", 
                   period="5y",
                   group_by='ticker')

df_temp = data.stack(level=0).reset_index()

df = pd.merge(df_temp, sector_data, how='left', on='Ticker')

df = df.sort_values(by=['Ticker', 'Date'])

# List of stocks for the user to select from
tickers = df['Ticker'].unique()

# User selects a stock ticker
selected_ticker = st.selectbox("Select a stock ticker:", tickers)

# Use the Pandas pct_change() function to calculate the change of the Adjusted closing price between each day 
df['Daily_Return'] = df['Adj Close'].pct_change()

# Calculate volatility (standard deviation of daily returns)
df['Volatility'] = df['Daily_Return'].rolling(window=30).std()

# Z-scores - for outlier detection
df['Z_Score_ACP'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: stats.zscore(x))
df['Z_Score_Volume'] = df.groupby('Ticker')['Volume'].transform(lambda x: stats.zscore(x))

# Define a function to calculate Bollinger Bands for a given window to help guage the volatility of a stock
def calculate_bollinger_bands(df, window):
    df[f'BB_upper_{window}'] = df[f'RM_{window}'] + (df['Adj Close'].rolling(window=window).std() * 2)
    df[f'BB_lower_{window}'] = df[f'RM_{window}'] - (df['Adj Close'].rolling(window=window).std() * 2)
    return df

for window in [30, 60, 90]:
    # Rolling Mean 
    df[f'RM_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=window).mean())
    # Rolling Standard Deviation
    df[f'RSTD_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=window).std())
    # Bollinger Bands 
    df = df.groupby('Ticker', group_keys=False).apply(lambda x: calculate_bollinger_bands(x, window)).reset_index(drop=True)

# RSI (Relative Strength Index) - momentum indicator that measures the speed and change of price movements
def calculate_rsi(df, window):
    # Calculate the difference in price from the previous step
    delta = df.diff(1)
    
    # Make two series: one for gains and one for losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate the rolling average of gains and losses
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI for different window sizes and add them to the dataframe
for window in [30, 60, 90]:
    df[f'RSI_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: calculate_rsi(x, window))
    
df.dropna(inplace=True)
# Initialize LightGBM model
train, test = train_test_split(df, test_size=0.2, shuffle=False)

# Define feature columns
feature_columns = [
    'Daily_Return', 'Volatility', 'Z_Score_ACP', 'Z_Score_Volume',
    'RM_30', 'RSTD_30', 'BB_upper_30', 'BB_lower_30',
    'RM_60', 'RSTD_60', 'BB_upper_60', 'BB_lower_60',
    'RM_90', 'RSTD_90', 'BB_upper_90', 'BB_lower_90',
    'RSI_30', 'RSI_60', 'RSI_90'
]

# Define the target column
target_column = 'Adj Close'

# Train and test splits (assuming you've already done this)
X_train = train[feature_columns]
y_train = train[target_column]

X_test = test[feature_columns]
y_test = test[target_column]

lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the model
lgb_model.fit(X_train, y_train)
joblib.dump(lgb_model, '/Users/sami/DSP/App/model/lgb_model2.pkl')