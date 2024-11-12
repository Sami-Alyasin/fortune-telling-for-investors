import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
import io

st.set_page_config(layout="wide",initial_sidebar_state="expanded")

# st.markdown("<h1 style='text-align: center;'>End-to-end Project Walkthrough</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Fortune Telling for Investors</h1>", unsafe_allow_html=True)

st.sidebar.markdown("Navigate to:", unsafe_allow_html=True)
st.sidebar.markdown("[Data Collection](#data-collection)", unsafe_allow_html=True)
st.sidebar.markdown("[Exploratory Data Analysis](#eda)", unsafe_allow_html=True)
st.sidebar.markdown("[Feature Engineering](#feature-engineering)", unsafe_allow_html=True)
st.sidebar.markdown("[Modeling](#modeling)", unsafe_allow_html=True)

st.divider()
left, middle, right = st.columns(3,gap='large')  
with left:
    st.page_link("Pages/Home.py",label = "About", icon = "📝")
    
with middle:
    st.page_link("Pages/Project.py",label = "Project walkthorugh", icon = "📚")  
    
with right:
    st.page_link("Pages/LiveModel.py",label = "Give the model a try!", icon = "🔮")

# add a disclaimer that this is in progress and not complete yet
st.warning('This page is a work in progress and is not complete yet. Please check the "In the works" section on the About page for more details.')

# Data Collection
with st.container(border=True):
    st.header('Data Collection',divider=True)

    st.markdown('''
                ### Download the Data from the Yahoo Finance API
                ''')

    st.code('''
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    import scipy.stats as stats
    ''')

    st.code('''
    # download the data for 30 companies from different sectors (Technology, Finance, Healthcare, Consumer Goods, Energy, Industrials)
    # we'll start with 5 years of data with an API call to Yahoo Finance
    data5Y = yf.download("AAPL MSFT GOOGL META AMZN JPM BAC WFC GS MS JNJ PFE MRK ABT BMY PG KO PEP NKE UL XOM CVX SLB COP BP BA CAT MMM GE HON", 
                    period="5y",
                    group_by='ticker')
    
    # take a look at the first 10 rows in the dataframe
    data5Y.head(10)
    data5Y.to_csv('/Users/sami/DSP/App/data/data5Y.csv')
    ''')

    url1 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/data5Y.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url1)
    df1 = load_data()
    # only show the first 10 rows of the data
    st.dataframe(df1.head(10))
    
    st.code('''
            # Reshape the DataFrame
    df_temp = data.stack(level=0).reset_index()
    df_temp.to_csv('stock_data_initial_t.csv', index=False)
    df_temp.head(10)
    ''')

    url2 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/stock_data_initial.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url2)
    df2 = load_data()
    st.dataframe(df2.head(10))

    st.code('''
            
    # Create a list of tickers
    tickers_list = "AAPL MSFT GOOGL META AMZN JPM BAC WFC GS MS JNJ PFE MRK ABT BMY PG KO PEP NKE UL XOM CVX SLB COP BP BA CAT MMM GE HON"
    tickers = tickers_list.split()

    # Create a list to hold our data that will be converted to a Dataframe later on
    data = []

    # Loop through the tickers, append data to our list, and finally create a Dataframe using the list  
    for ticker in tickers:
        # for some reason the .info method was not working for Caterpillar Inc., so I had to hardcode the sector and company name
        if ticker == 'CAT':
            sector = 'Industrials'
            c_name = 'Caterpillar Inc.'
        else:
            sector = yf.Ticker(ticker).info['sector']
            c_name = yf.Ticker(ticker).info['longName']
        data.append({'Ticker': ticker,
                    'Sector': sector,
                    'Company Name': c_name
                    })

    ind_sec_df = pd.DataFrame(data)

    ind_sec_df.to_csv('sector_data.csv', index=False)

    ind_sec_df
    ''')
    
    url3 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/sector_data.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url3)
    df3 = load_data()
    st.dataframe(df3)

    st.code('''
            # Merge the Dataframes
    df = pd.merge(df_temp, ind_sec_df, how='left', on='Ticker')
    df.to_csv('stock_data_initial.csv', index=False)
    df.head(10)
    ''')

    url4 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/stock_data_merged.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url4)
    df4 = load_data()
    st.dataframe(df4.head(10))

# EDA
with st.container(border=True):
    st.header('EDA',divider=True)
    st.markdown('''
                ### Exploratory Data Analysis
                ''')

    st.code('''
            df.info()
    ''')

    buffer = io.StringIO()
    df4.info(buf=buffer)
    s = buffer.getvalue()

    st.text(s)

    st.code('''
                df.describe()
                ''')

    st.write(df4.describe())

    st.code('''
            # Remove Timezone from Date to make it easier to work with - we only have one timezone in the data 'UTC' so it is ok to remove
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    ''')

    df4['Date'] = pd.to_datetime(df4['Date']).dt.tz_localize(None)

    st.code('''
            # Boxplot for closing prices by sector
    plt.figure(figsize=(20, 8))
    sns.boxplot(y='Sector', x='Adj Close', data=df, palette="Set3", hue='Sector')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Stock Prices by Sector', fontsize=16, fontweight='bold')
    plt.xlabel('Adjusted Closing Price', fontsize=14)
    plt.ylabel('Sector', fontsize=14)
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, linestyle='--', alpha=0.7)
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/box%20plot%20by%20sector.png')

    st.code('''
            # Group by sector and calculate mean closing prices
    sector_performance = df.groupby('Sector')['Adj Close'].mean().sort_values(ascending=False)

    # Bar plot to visualize sector performance
    ax = sector_performance.plot(kind='bar', figsize=(20, 8), zorder=3)

    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Sector', fontsize=14)
    plt.ylabel('Average Adjusted Closing Price', fontsize=14)
    plt.title('Average Stock Prices by Sector', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)  
    plt.gca().set_facecolor('#f0f0f0')

    bars = ax.patches
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_zorder(3)
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/Average%20Stock%20Price%20by%20sector.png')

    st.code('''
    # Below, I'm using visual methods to explore the distribution of the adjusted closing prices of the stocks

    # Assign the number of columns and rows. We'll have one row per ticker and we'll use two columns to display a histogram and a box plot
    num_cols = 2
    num_rows = len(tickers)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    fig.tight_layout(pad=5.0)

    # Make it look nice :)
    sns.set_style("whitegrid")
    color_palette = sns.color_palette("viridis", as_cmap=True)

    # Loop through each ticker and create two plots for each: histogram and box plot
    for i, tkr in enumerate(tickers):
        # Histogram for 'Adj Close' on the left (column 0)
            # Histogram with KDE (Kernel Density Estimate): A histogram will give you a visual sense of the distribution. 
            # You can overlay a KDE plot to see if the distribution resembles a normal (bell-shaped) curve.
        sns.histplot(
            df[df['Ticker'] == tkr]['Adj Close'], 
            bins=50, 
            kde=True, 
            color=color_palette(i / len(tickers)), 
            ax=axes[i, 0]
            )  
        c_name = df[df['Ticker'] == tkr]['Company Name'].iloc[0]
        axes[i, 0].set_title(f'{c_name}  [{tkr}]', fontsize=14, fontweight='bold')
        axes[i, 0].set_xlabel('Adjusted Closing Price', fontsize=12)
        axes[i, 0].set_ylabel('Frequency', fontsize=12)
        
        # Box plot for 'Adj Close' on the middle (column 1)
        sns.boxplot(
            x=df[df['Ticker'] == tkr]['Adj Close'], 
            color=color_palette(i / len(tickers)), 
            ax=axes[i, 1]
            )  
        axes[i, 1].set_title(f'{c_name}  [{tkr}]', fontsize=14, fontweight='bold')
        axes[i, 1].set_xlabel('Adjusted Closing Price', fontsize=12)
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/distribution.png')

    st.code('''
    # We can also use the following statistical methods to test for normality:
    # 1. Shapiro-Wilk Test
    # 2. Kolmogorov-Smirnov Test

    # We'll use the shapiro test

    from scipy.stats import shapiro

    # BEGIN: Shapiro-Wilk test for normality
    p_values = []

    for ticker in tickers:
        stat, p_value = shapiro(df[df['Ticker'] == ticker]['Adj Close'])
        p_values.append({
            'Ticker': ticker, 
            'P_value': p_value
            })

    temp_df = pd.DataFrame(p_values)
    temp_df['Normal Distribution'] = temp_df['P_value'].apply(lambda x: 'Pass' if x > 0.05 else 'Fail')

    temp_df
    ''')

    url5 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/normality_test.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url5)
    df5 = load_data()
    df5['P_value'] = df5['P_value'].apply(lambda x: "{:.4f}".format(x))
    st.dataframe(df5)
        
# Feature Engineering
with st.container(border=True):
    st.header('Feature Engineering',divider=True)

    st.code('''
            # Calculate daily returns & Volatility

    # Start by sorting the data by Ticker and Date
    df = df.sort_values(by=['Ticker', 'Date'])

    # Use the Pandas pct_change() function to calculate the change of the Adjusted closing price between each day 
    df['Daily_Return'] = df['Adj Close'].pct_change()

    # Calculate volatility (standard deviation of daily returns)
    df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()
    df['Volatility_15'] = df['Daily_Return'].rolling(window=15).std()
    df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
    ''')

    st.code('''
    # Assign the number of columns and rows. 
    # We'll have one row per ticker and we'll use two columns to display a time series for the adjusted closing price and the volatility 
    num_cols = 2
    num_rows = len(tickers)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    fig.tight_layout(pad=5.0)

    # Make it look nice :)
    sns.set_style("whitegrid") 
    color_palette = sns.color_palette("viridis", as_cmap=True)

    # Specify start date for the volatility plot to start 2 months after the data start date
    # Removes unwanted noise since we're using the standard deviation of a 30 day rolling window
    start_date = df['Date'].min() + pd.DateOffset(months=2)

    # Loop through each ticker and create two plots for each: histogram and box plot
    for i, tkr in enumerate(tickers):
        # Line plot for 'Adj Close' on the left (column 0)
        sns.lineplot(
            data=df[df['Ticker'] == tkr], 
            x='Date', 
            y='Adj Close', 
            label='Adj Close', 
            color=color_palette(i / len(tickers)), 
            ax=axes[i, 0]
            )
        c_name = df[df['Ticker'] == tkr]['Company Name'].iloc[0]
        axes[i, 0].set_title(f'{c_name} - Adjusted Closing Price', fontsize=14, fontweight='bold')
        axes[i, 0].set_xlabel('Date', fontsize=12)
        axes[i, 0].set_ylabel('Adjusted Closing Price', fontsize=12)
        
        # Line plot for 'Volatility' on the right (column 1)
        sns.lineplot(
            data=df[(df['Ticker'] == tkr) & (df['Date'] >= start_date)],
            x='Date', 
            y='Volatility_30', 
            label='Volatility', 
            color=color_palette(i / len(tickers)), 
            ax=axes[i, 1]
            )
        c_name = df[df['Ticker'] == tkr]['Company Name'].iloc[0]
        axes[i, 1].set_title(f'{c_name} - Volatility', fontsize=14, fontweight='bold')
        axes[i, 1].set_xlabel('Date', fontsize=12)
        axes[i, 1].set_ylabel('Volatility', fontsize=12)   
        ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/volatility.png')

    st.code('''
            # Let's add some helpful features to the data
    from scipy import stats

    # Z-scores - for outlier detection
    df['Z_Score_ACP'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: stats.zscore(x))
    df['Z_Score_Volume'] = df.groupby('Ticker')['Volume'].transform(lambda x: stats.zscore(x))

    # Define a function to calculate Bollinger Bands for a given window to help guage the volatility of a stock
    def calculate_bollinger_bands(df, window):
        df[f'BB_upper_{window}'] = df[f'RM_{window}'] + (df['Adj Close'].rolling(window=window).std() * 2)
        df[f'BB_lower_{window}'] = df[f'RM_{window}'] - (df['Adj Close'].rolling(window=window).std() * 2)
        return df

    # Calculate Bollinger Bands for different windows
    for window in [5,10,15,30,60,90]:
        # Rolling Mean 
        df[f'RM_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=window).mean())
        # Rolling Standard Deviation
        df[f'RSTD_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=window).std())
        # Bollinger Bands 
        df = df.groupby('Ticker', group_keys=False).apply(lambda x: calculate_bollinger_bands(x, window)).reset_index(drop=True)
    df
    ''')

    st.code('''
            # RSI (Relative Strength Index) - momentum indicator that measures the speed and change of price movements
    def calculate_rsi(data, window):
        # Calculate the difference in price from the previous step
        delta = data.diff(1)
        
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
    for window in [5,10,15,30,60,90]:
        df[f'RSI_{window}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: calculate_rsi(x, window))
    ''')

    st.code('''
    # Since the rolling window features have created nulls, we will drop the records that have null values, that way we can have a clean dataset
    df.dropna(inplace=True)

    # Export the data to a CSV file for next steps - I don't want to re-run the code above every time I need the data
    # I will be uploading this to the Github repo
    df.to_csv('stock_data.csv', index=False)
    ''')

# Modeling
with st.container(border=True):
    st.header('Modeling',divider=True)

    st.markdown('''
    ### We will implement and evaulate the following models:
    - **ARIMA (AutoRegressive Integrated Moving Average)**
    - **Random Forest Regressor**
    - **XGBoost**
    - **LightGBM**
    - **GRU (Gated Recurrent Unit)**
    ''')
# ARIMA
    st.markdown('''
    ### ARIMA
    ''')
    st.code('''
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score

    # get historical data for APPL for the last 20 years
    data_ARIMA = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)

    # Filtering data for a single stock to implement ARIMA model
    stock_ticker = 'AAPL'
    stock_data = data_ARIMA[data_ARIMA['Ticker'] == stock_ticker]['Adj Close']

    # Split the data into training and test sets
    train_data, test_data = train_test_split(stock_data, test_size=0.2, shuffle=False)

    # Differencing to make time series stationary
    differenced_data = train_data.diff().dropna()

    # Augmented Dickey-Fuller test to check for stationarity
    from statsmodels.tsa.stattools import adfuller
    result1 = adfuller(train_data)
    print(f'Training data ADF Statistic: {result1[0]}')
    print(f'Training data p-value: {result1[1]}')
    print("Original data is stationary" if result1[1] <= 0.05 else "Original data is not stationary, differencing required")

    result2 = adfuller(differenced_data)
    print(f'Differenced data ADF Statistic: {result2[0]}')
    print(f'Differenced data p-value: {result2[1]}')
    print("Differenced data is stationary" if result2[1] <= 0.05 else "Differenced data is not stationary")

    # Finding the order (p, d, q) using ACF and PACF plots
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    sm.graphics.tsa.plot_acf(differenced_data, ax=ax[0])
    sm.graphics.tsa.plot_pacf(differenced_data, ax=ax[1])
    plt.show()
    ''')
    
    st.text('''
            Training data ADF Statistic: -2.3613780288367243
            Training data p-value: 0.1529351125960524
            Original data is not stationary, differencing required

            Differenced data ADF Statistic: -31.084016077072704
            Differenced data p-value: 0.0
            Differenced data is stationary
            ''')
    
    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/acf_pacf.png')
    
    st.markdown('''
    ### AFD, ACF, & PACF Interpretation

    - **ADF (Augmented Dickey-Fuller) test:** 
        - The ADF test we ran suggests that our original data is not stationary (p-value: 0.1529), meaning it likely has trends or other non-stationary components that ARIMA cannot handle well directly, and that the differenced data is stationary (p-value: 0.0).
        - The ADF Statistic value of -31.084 in the differenced data is a very large negative number, which indicates that the series is highly stationary.
    - **ACF Plot (Autocorrelation Function):**
        - The ACF plot shows the correlation of the time series with its past values (lags).
        - The significant spike at **lag 0** is expected since a time series is always perfectly correlated with itself.
        - The key aspect to look for in this plot is the **cut-off point** (where correlations drop within the blue confidence interval bands). A cut-off after lag 1 or 2 could indicate the Moving Average (`q`) component.
        - In our plot, there is a significant spike at **lag 0**, and all subsequent lags fall within the confidence interval, which suggests a cut-off point. This behavior indicates that **`q` should be 0 or 1**.
    - **PACF Plot (Partial Autocorrelation Function):**
        - The PACF plot shows the partial correlation of the time series with its own lagged values, after removing the effect of intermediate lags.
        - The **cut-off point** in the PACF plot helps identify the order of the Autoregressive (`p`) component.
        - In our plot, we see a significant spike at **lag 0**, followed by most subsequent lags falling within the confidence bands. This suggests that **`p` should also be around 0 or 1**.
                ''')
   
    st.code('''
            # Fit ARIMA model
            p, d, q = 1, 2, 1
            model = sm.tsa.ARIMA(train_data, order=(p, d, q))
            model_fit = model.fit()

            # Model Summary
            print(model_fit.summary())

            # Predicting the next day's price using get_forecast
            forecast_object = model_fit.get_forecast(steps=len(test_data))
            forecast = forecast_object.predicted_mean
            conf_int = forecast_object.conf_int()

            # Plotting actual vs predicted
            plt.figure(figsize=(20, 8))
            plt.plot(test_data, label='Actual Price')
            plt.plot(test_data.index, forecast, label='Predicted Price', color='red')
            plt.xlabel('Date')
            plt.ylabel('Adjusted Close Price')
            plt.title(f'Actual vs Predicted Values for {stock_ticker} Stock Price')
            plt.legend()
            plt.show()

            # Evaluating Model Performance
            y_true = test_data
            y_pred = forecast
            arima_mae = mean_absolute_error(y_true, y_pred)
            arima_rmse = root_mean_squared_error(y_true, y_pred)
            arima_mse = np.square(arima_rmse)
            arima_r2 = r2_score(y_true, y_pred)

            print(f'MAE: {arima_mae}')
            print(f'MSE: {arima_mse}')
            print(f'RMSE: {arima_rmse}')
            print(f'R2 Score: {arima_r2}')

            ''') 
    
    st.text('''
                                           SARIMAX Results                                
        ==============================================================================
        Dep. Variable:              Adj Close   No. Observations:                  936
        Model:                 ARIMA(1, 2, 1)   Log Likelihood               -2223.847
        Date:                Mon, 11 Nov 2024   AIC                           4453.694
        Time:                        17:10:36   BIC                           4468.213
        Sample:                             0   HQIC                          4459.231
                                        - 936                                         
        Covariance Type:                  opg                                         
        ==============================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
        ------------------------------------------------------------------------------
        ar.L1         -0.0169      0.029     -0.586      0.558      -0.074       0.040
        ma.L1         -0.9999      0.130     -7.668      0.000      -1.256      -0.744
        sigma2         6.7992      0.907      7.494      0.000       5.021       8.578
        ===================================================================================
        Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):               109.34
        Prob(Q):                              0.94   Prob(JB):                         0.00
        Heteroskedasticity (H):               1.19   Skew:                            -0.05
        Prob(H) (two-sided):                  0.12   Kurtosis:                         4.67
        ===================================================================================
        
        
                ''')
    
    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/ARIMA.png')
    
    st.text('''
            MAE: 13.31685522450983
            MSE: 298.6497810176591
            RMSE: 17.281486655310044
            R2 Score: 0.39032436821929584
                        ''')
    
    st.markdown('''
        #### Now that we've demonstarted how we're able to implement and evaluate an ARIMA model for a single stock, we will create a function that will run this process for the rest of the stocks (30 in total) and take their average evaluation metrics to compare with the other models we will implement.            
                ''')

    st.code('''
            # Function to fit and evaluate ARIMA model
    def ARIMA_model(ticker, p, d, q):
        stock_data = data[data['Ticker'] == ticker]['Adj Close']

        # Split the data into training and test sets
        train_data, test_data = train_test_split(stock_data, test_size=0.2, shuffle=False)
        
        # Fit the model
        model = sm.tsa.ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        
        # Predicting the next day's price using get_forecast
        forecast_object = model_fit.get_forecast(steps=len(test_data))
        forecast = forecast_object.predicted_mean
        
        # Evaluating Model Performance
        y_true = test_data
        y_pred = forecast
        arima_mae = mean_absolute_error(y_true, y_pred)
        arima_rmse = root_mean_squared_error(y_true, y_pred)
        arima_mse = mean_squared_error(y_true, y_pred)
        arima_r2 = r2_score(y_true, y_pred)
        
        return arima_mae, arima_rmse, arima_mse, arima_r2

    # Running ARIMA model for 30 different tickers and calculating average metrics
    tickers = data['Ticker'].unique()[:30]
    metrics = {'mae': [], 'rmse': [], 'mse': [], 'r2': []}

    for ticker in tickers:
        try:
            arima_mae, arima_rmse, arima_mse, arima_r2 = ARIMA_model(ticker, 1, 2, 1)
            metrics['mae'].append(arima_mae)
            metrics['rmse'].append(arima_rmse)
            metrics['mse'].append(arima_mse)
            metrics['r2'].append(arima_r2)
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
            continue

    # Calculating average metrics across all tickers
    arima_avg_mae = np.mean(metrics['mae'])
    arima_avg_rmse = np.mean(metrics['rmse'])
    arima_avg_mse = np.mean(metrics['mse'])
    arima_avg_r2 = np.mean(metrics['r2'])
            ''')
    
    st.code('''
            # Create a DataFrame to store the evaluation metrics
            evaluation_df = pd.DataFrame({
                'Model': [],
                'MAE': [],
                'MSE': [],
                'RMSE': [],
                'R2': []
            })

            # append ARIMA evaluation metrics to the DataFrame
            if 'ARIMA' in evaluation_df['Model'].values:
                evaluation_df.loc[evaluation_df['Model'] == 'ARIMA', ['MAE', 'MSE', 'RMSE', 'R2']] = [arima_avg_mae, arima_avg_mse, arima_avg_rmse, arima_avg_r2]
            else:
                evaluation_df.loc[len(evaluation_df)] = ['ARIMA', arima_avg_mae, arima_avg_mse, arima_avg_rmse, arima_avg_r2]
                
            evaluation_df
    ''')

    url6 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/evaluation_df1.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url6)
    df6 = load_data()
    st.dataframe(df6)  

# Random Forest Regressor
    st.markdown('''
    ### Random Forest Regressor
    Data Preparation:
    We already have our data with features like rolling means, standard deviations, Bollinger Bands, RSI, etc. The next step is to ensure our dataset is ready for modeling.

    Create Feature and Target Data:
    Separate the features from our target (Adj Close or Daily Return) for both the training and test sets.
    ''')

    st.code('''
        # Prepare the data for RandomForestRegressor
        # Load data
        data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)

        # Split data into train and test sets
        train, test = train_test_split(data, test_size=0.2, shuffle=False)

        # Define feature columns
        feature_columns = [
            'Daily_Return', 'Volatility_5', 'Volatility_15', 'Volatility_30', 'Z_Score_ACP', 'Z_Score_Volume',
            'RM_5', 'RSTD_5', 'BB_upper_5', 'BB_lower_5',
            'RM_10', 'RSTD_10', 'BB_upper_10', 'BB_lower_10',
            'RM_15', 'RSTD_15', 'BB_upper_15', 'BB_lower_15',
            'RSI_5', 'RSI_10', 'RSI_15',
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
    ''')

    st.markdown('''
    Fit the RandomForestRegressor with All Features
    ''')

    st.code('''
            from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    import numpy as np

    # Initialize RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf_model.fit(X_train, y_train)

    # Predict on the test data
    rf_predictions = rf_model.predict(X_test)

    # Evaluate the model
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_rmse = root_mean_squared_error(y_test, rf_predictions)
    rf_mse = np.square(rf_rmse)
    rf_r2 = r2_score(y_test, rf_predictions)

    # Check if the model name already exists in the DataFrame
    if 'RandomForest' in evaluation_df['Model'].values:
        evaluation_df.loc[evaluation_df['Model'] == 'RandomForest', ['MAE', 'MSE', 'RMSE', 'R2']] = [rf_mae, rf_mse, rf_rmse, rf_r2]
    else:
        # Append RandomForestRegressor results
        evaluation_df.loc[len(evaluation_df)] = ['RandomForest', rf_mae, rf_mse, rf_rmse, rf_r2]

    evaluation_df
    ''')
    url7 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/evaluation_df2.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url7)
    df7 = load_data()
    st.dataframe(df7) 


    st.code('''
    import matplotlib.pyplot as plt

    # Predict on the test data using the Random Forest model
    rf_predictions = rf_model.predict(X_test)

    # Plot the actual vs predicted prices
    plt.figure(figsize=(20, 6))
    plt.plot(y_test.values, color='blue', label='Actual Stock Price')
    plt.plot(rf_predictions, color='red', label='Predicted Stock Price')
    plt.title('Random Forest Model - Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.gca().set_facecolor('#f0f0f0')  # Set background color
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/rf.png')

    st.markdown('''
    Feature Importance Analysis:
    To understand which features are most important for the RandomForest model, we can use the feature importance method provided by the model.

    The feature importance plot will help us identify the most predictive features for this model. Features with very low importance can potentially be removed to simplify the model.
    ''')

    st.code('''
    # Get feature importances
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', dodge=False, legend=False)
    plt.title('Feature Importance from Random Forest Regressor', fontsize=16, fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, linestyle='--', alpha=0.7)
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/featureimportance1.png')

    st.markdown('''
    Recursive Feature Elimination (RFE):
    Now, let's apply Recursive Feature Elimination (RFE) to select the most important features step by step.

    This will output the rank of each feature, where a ranking of 1 indicates that the feature was selected as one of the most important.
    ''')

    st.code('''
    from sklearn.feature_selection import RFE

    # Initialize RFE with RandomForestRegressor as the estimator
    rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=10)

    # Fit RFE
    rfe.fit(X_train, y_train)

    # Get the ranking of features
    rfe_ranking = pd.DataFrame({
        'Feature': feature_columns,
        'Ranking': rfe.ranking_
    }).sort_values(by='Ranking')

    print(rfe_ranking)
    ''')

    url8 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/rfe_ranking.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url8)
    df8 = load_data()
    st.dataframe(df8) 

    st.markdown('''
    Re-train the Model with Selected Features. Since we identified the top features from RFE & feature importance, we'll retrain our model using only the selected features.
    ''')

    st.code('''
    # Select top features (from RFE or feature importance analysis)
    selected_features = rfe_ranking[rfe_ranking['Ranking'] == 1]['Feature'].tolist()

    # Train the model with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Refit the model
    rf_model.fit(X_train_selected, y_train)

    # Predict on the test set
    rf_predictions_selected = rf_model.predict(X_test_selected)

    # Evaluate the model
    rf_mae_selected = mean_absolute_error(y_test, rf_predictions_selected)
    rf_rmse_selected = root_mean_squared_error(y_test, rf_predictions_selected)
    rf_mse_selected = np.square(rf_rmse_selected)
    rf_r2_selected = r2_score(y_test, rf_predictions_selected)

    # Append Optimized RandomForestRegressor results
    # Check if the model name already exists in the DataFrame
    model_name = 'RandomForest w/ RFE'
    if model_name in evaluation_df['Model'].values:
        evaluation_df.loc[evaluation_df['Model'] == model_name, ['MAE', 'MSE', 'RMSE', 'R2']] = [rf_mae_selected, rf_mse_selected, rf_rmse_selected, rf_r2_selected]
    else:
        evaluation_df.loc[len(evaluation_df)] = [model_name, rf_mae_selected, rf_mse_selected, rf_rmse_selected, rf_r2_selected]

    evaluation_df
    ''')
    url9 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/evaluation_df3.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url9)
    df9 = load_data()
    st.dataframe(df9)


    st.markdown('''
    Now we'll apply cross-validation using TimeSeriesSplit to verify the generalization ability of the model.
    ''')

    st.code('''
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    cv_scores_w_rfe = cross_val_score(rf_model, X_train_selected, y_train, cv=tscv, scoring='neg_mean_squared_error')

    # Convert negative MSE to positive RMSE
    cv_rmse = np.sqrt(-cv_scores)
    cv_rmse_w_rfe = np.sqrt(-cv_scores_w_rfe)

    print(f"Cross-validated RMSE scores without RFE: {cv_rmse}")
    print(f"Mean RMSE: {cv_rmse.mean()}")

    print(f"Cross-validated RMSE scores with RFE: {cv_rmse_w_rfe}")
    print(f"Mean RMSE with RFE: {cv_rmse_w_rfe.mean()}")
    ''')

    st.write('Cross-validated RMSE scores without RFE: [32.63447453  2.22874748 27.71891349  4.43199662  6.10554772]')
    st.write('Mean RMSE: 14.62393596757805')
    st.write('Cross-validated RMSE scores with RFE: [30.26314701  2.22361474 28.32227043  4.59673514  5.93427474]')
    st.write('Mean RMSE with RFE: 14.26800841253872')


    st.markdown('''
    They both generalize well, and the model with RFE performs slightly better than the model without RFE.
    ''')

    st.markdown('''
    ### XGBoost
    ''')

    st.code('''
            import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    # Initialize XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

    # Fit the model
    xgb_model.fit(X_train, y_train)

    # Predict on the test data
    xgb_predictions = xgb_model.predict(X_test)

    # Evaluate the model
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_rmse = root_mean_squared_error(y_test, xgb_predictions)
    xgb_mse = np.square(xgb_rmse)
    xgb_r2 = r2_score(y_test, xgb_predictions)

    # append XGBoost results
    model_name = 'XGBoost'
    if model_name in evaluation_df['Model'].values:
        evaluation_df.loc[evaluation_df['Model'] == model_name, ['MAE', 'MSE', 'RMSE', 'R2']] = [xgb_mae, xgb_mse, xgb_rmse, xgb_r2]
    else:
        evaluation_df.loc[len(evaluation_df)] = [model_name, xgb_mae, xgb_mse, xgb_rmse, xgb_r2]

    evaluation_df
    ''')

    url10 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/evaluation_df4.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url10)
    df10 = load_data()
    st.dataframe(df10)

    st.code('''
    import matplotlib.pyplot as plt

    # Predict on the test data using the XGBoost model
    xgb_predictions = xgb_model.predict(X_test)

    # Plot the actual vs predicted prices
    plt.figure(figsize=(20, 6))
    plt.plot(y_test.values, color='blue', label='Actual Stock Price')
    plt.plot(xgb_predictions, color='red', label='Predicted Stock Price')
    plt.title('XGBoost Model - Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.gca().set_facecolor('#f0f0f0')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/XGB.png')

    st.code('''
            # Get feature importances
    xgb_feature_importance = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': xgb_feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Plot all the feature importances
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', dodge=False, legend=False)
    plt.title('Top 10 Feature Importance from XGBoost', fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, linestyle='--', alpha=0.7)
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/featureimportance2.png')

    st.code('''
    # Initialze RFE with XGBoost as the estimator
    rfe_xgb = RFE(estimator=xgb_model, n_features_to_select=10)

    # Fit RFE
    rfe_xgb.fit(X_train, y_train)

    # Get the ranking of features
    rfe_xgb_ranking = pd.DataFrame({
        'Feature': feature_columns,
        'Ranking': rfe_xgb.ranking_
    }).sort_values(by='Ranking')

    print(rfe_xgb_ranking)
    ''')
    
    url10a = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/rfe_xgb_ranking.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url10a)
    df10a = load_data()
    st.dataframe(df10a)
    
    st.code('''
    # Select top features (from RFE or feature importance analysis)
    selected_features = rfe_xgb_ranking[rfe_xgb_ranking['Ranking'] == 1]['Feature'].tolist()

    # Train the model with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Initialize XGBoost model
    xgb_model2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

    # Fit the model
    xgb_model2.fit(X_train_selected, y_train)

    # Predict on the test data
    xgb_predictions = xgb_model2.predict(X_test_selected)

    # Evaluate the model
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_rmse = root_mean_squared_error(y_test, xgb_predictions)
    xgb_mse = np.square(xgb_rmse)
    xgb_r2 = r2_score(y_test, xgb_predictions)

    # append XGBoost results
    model_name = 'XGBoost w/RFE'
    if model_name in evaluation_df['Model'].values:
        evaluation_df.loc[evaluation_df['Model'] == model_name, ['MAE', 'MSE', 'RMSE', 'R2']] = [xgb_mae, xgb_mse, xgb_rmse, xgb_r2]
    else:
        evaluation_df.loc[len(evaluation_df)] = [model_name, xgb_mae, xgb_mse, xgb_rmse, xgb_r2]
    evaluation_df.to_csv('/Users/sami/DSP/App-SPP/data/evaluation_df4b.csv', index=False)
    evaluation_df
            ''')
    
    url10b = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/evaluation_df4b.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url10b)
    df10b = load_data()
    st.dataframe(df10b)
    
    st.markdown('''
    Cross-Validation with XGBoost
    Just like with RandomForest, you can apply TimeSeriesSplit to perform cross-validation with XGBoost
    ''')

    st.code('''
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform cross-validation
    cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')

    # Convert negative MSE to positive RMSE
    cv_rmse_xgb = np.sqrt(-cv_scores_xgb)
    print(f"XGBoost Cross-validated RMSE scores: {cv_rmse_xgb}")
    print(f"Mean RMSE: {cv_rmse_xgb.mean()}")
    ''')

    st.write('XGBoost Cross-validated RMSE scores: [44.56418014  5.39636316 35.16814353  8.99874153  8.01419586]')
    st.write('Mean RMSE: 20.428324843656085')


    st.markdown('''
    ### LightGBM
    ''')

    st.code('''
    import lightgbm as lgb

    # Initialize LightGBM model
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Fit the model
    lgb_model.fit(X_train, y_train)

    # Predict on the test data
    lgb_predictions = lgb_model.predict(X_test)

    # Evaluate the model
    lgb_mae = mean_absolute_error(y_test, lgb_predictions)
    lgb_rmse = root_mean_squared_error(y_test, lgb_predictions)
    lgb_mse = np.square(lgb_rmse)
    lbg_r2 = r2_score(y_test, lgb_predictions)

    # Append LightGBM results
    model_name = 'LightGBM'
    if model_name in evaluation_df['Model'].values:
        evaluation_df.loc[evaluation_df['Model'] == model_name, ['MAE', 'MSE', 'RMSE', 'R2']] = [lgb_mae, lgb_mse, lgb_rmse, lbg_r2]
    else:
        evaluation_df.loc[len(evaluation_df)] = [model_name, lgb_mae, lgb_mse, lgb_rmse, lbg_r2]
    evaluation_df
    ''')

    url11 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/evaluation_df5.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url11)
    df11 = load_data()
    st.dataframe(df11)
    
    # evaluation_df5 = pd.read_csv('/Users/sami/DSP/App/data/evaluation_df5.csv')
    # st.dataframe(data=evaluation_df5)

    st.code('''
    import matplotlib.pyplot as plt

    # Predict on the test data using the LightGBM model
    lgb_predictions = lgb_model.predict(X_test)

    # Plot the actual vs predicted prices
    plt.figure(figsize=(20, 6))
    plt.plot(y_test.values, color='blue', label='Actual Stock Price')
    plt.plot(lgb_predictions, color='red', label='Predicted Stock Price')
    plt.title('LightGBM Model - Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.gca().set_facecolor('#f0f0f0')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/LGBM.png')

    st.code('''
    # Get feature importances
    lgbm_feature_importance = lgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': lgbm_feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Plot all the feature importances
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', dodge=False, legend=False)
    plt.title('Feature Importance from LightGBM', fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, linestyle='--', alpha=0.7)
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/featureimportance3.png')


    st.markdown('''
    Cross-Validation with LightGBM
    ''')

    st.code('''
    # Perform cross-validation
    cv_scores_lgb = cross_val_score(lgb_model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')

    # Convert negative MSE to positive RMSE
    cv_rmse_lgb = np.sqrt(-cv_scores_lgb)
    print(f"LightGBM Cross-validated RMSE scores: {cv_rmse_lgb}")
    print(f"Mean RMSE: {cv_rmse_lgb.mean()}")
    ''')

    st.write('LightGBM Cross-validated RMSE scores: [44.54512687  4.63374326 32.73412673  9.54476619  7.68359448]')
    st.write('Mean RMSE: 19.828271506889116')

    st.markdown('''
    ### Hyperparameter tuning
    ''')

    st.code('''
    from sklearn.model_selection import RandomizedSearchCV
    # set up the parameter grid for XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5]
    }
    # set up the parameter grid for LightGBM
    lgb_param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [20, 31, 50, 70],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.2],
        'reg_lambda': [0, 0.1, 0.2]
    }
    ''')

    st.markdown('''
    Set Up RandomizedSearchCV:
    RandomizedSearchCV will randomly sample from the parameter grid a specified number of times, which speeds up the search process.
    ''')

    st.code('''
    xgb_random_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_distributions=xgb_param_grid,
        n_iter=50,
        scoring='neg_mean_squared_error',
        cv=TimeSeriesSplit(n_splits=5),
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    xgb_random_search.fit(X_train, y_train)

    # Best parameters and score
    print("Best Parameters for XGBoost:", xgb_random_search.best_params_)
    print("Best RMSE Score for XGBoost:", np.sqrt(-xgb_random_search.best_score_))
    ''')

    st.write('''Best Parameters for XGBoost: {'subsample': 0.8, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.8}''')
    st.write('Best RMSE Score for XGBoost: 22.48940465755753')


    st.code('''
    lgb_random_search = RandomizedSearchCV(
        estimator=lgb.LGBMRegressor(random_state=42),
        param_distributions=lgb_param_grid,
        n_iter=50, 
        scoring='neg_mean_squared_error',
        cv=TimeSeriesSplit(n_splits=5),
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    lgb_random_search.fit(X_train, y_train)

    # Best parameters and score
    print("Best Parameters for LightGBM:", lgb_random_search.best_params_)
    print("Best RMSE Score for LightGBM:", np.sqrt(-lgb_random_search.best_score_))
    ''')

    st.write('''Best Parameters for LightGBM: {'subsample': 1.0, 'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 20, 'n_estimators': 200, 'min_child_samples': 20, 'learning_rate': 0.1, 'colsample_bytree': 0.6}''')
    st.write('''Best RMSE Score for LightGBM: 21.276306743562923''')

    st.markdown('''
    Evaluate the best model after tunining the hyperparameters
    ''')

    st.code('''
    best_xgb_model = xgb_random_search.best_estimator_
    xgb_predictions = best_xgb_model.predict(X_test)

    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_rmse = root_mean_squared_error(y_test, xgb_predictions)
    xgb_mse = np.square(xgb_rmse)
    xgb_r2 = r2_score(y_test, xgb_predictions)

    # Append Tuned XGBoost results
    model_name = 'Tuned XGBoost'
    if model_name in evaluation_df['Model'].values:
        evaluation_df.loc[evaluation_df['Model'] == model_name, ['MAE', 'MSE', 'RMSE', 'R2']] = [xgb_mae, xgb_mse, xgb_rmse, xgb_r2]
    else:
        evaluation_df.loc[len(evaluation_df)] = [model_name, xgb_mae, xgb_mse, xgb_rmse, xgb_r2]
    evaluation_df

    ''')

    url12 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/evaluation_df6.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url12)
    df12 = load_data()
    st.dataframe(df12)

    st.code('''
    # Get feature importances
    tunedxgb_feature_importance = best_xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': tunedxgb_feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Plot all the feature importances
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', dodge=False, legend=False)
    plt.title('Feature Importance from Tuned XGBoost', fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, linestyle='--', alpha=0.7)
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/featureimportance4.png')

    st.code('''
    best_lgb_model = lgb_random_search.best_estimator_
    lgb_predictions = best_lgb_model.predict(X_test)

    lgb_mae = mean_absolute_error(y_test, lgb_predictions)
    lgb_rmse = root_mean_squared_error(y_test, lgb_predictions)
    lgb_mse = np.square(lgb_rmse)
    lgb_r2 = r2_score(y_test, lgb_predictions)

    # Append Tuned LightGBM results
    model_name = 'Tuned LightGBM'
    if model_name in evaluation_df['Model'].values:
        evaluation_df.loc[evaluation_df['Model'] == model_name, ['MAE', 'MSE', 'RMSE', 'R2']] = [lgb_mae, lgb_mse, lgb_rmse, lgb_r2]
    else:
        evaluation_df.loc[len(evaluation_df)] = [model_name, lgb_mae, lgb_mse, lgb_rmse, lgb_r2]
    evaluation_df
    ''')

    url13 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/evaluation_df7.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url13)
    df13 = load_data()
    st.dataframe(df13)

    st.code('''
    # Get feature importances
    tunedlbg_feature_importance = best_lgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': tunedlbg_feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Plot all the feature importances
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', dodge=False, legend=False)
    plt.title('Feature Importance from Tuned LightGBM', fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, linestyle='--', alpha=0.7)
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/featureimportance5.png')

    st.markdown('''
    ### GRU (Gated Recurrent Unit)
    ''')

    st.code('''
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    ''')

    st.code('''
    # Load data
    GRU_data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)
    GRU_data = GRU_data[['Adj Close']]  


    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(GRU_data)

    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    ''')


    st.code('''
    def create_sequences(GRU_data, sequence_length):
        X, y = [], []
        for i in range(sequence_length, len(GRU_data)):
            X.append(GRU_data[i-sequence_length:i, 0])
            y.append(GRU_data[i, 0])
        return np.array(X), np.array(y)

    sequence_length = 60 
    X_train_gru, y_train_gru = create_sequences(train_data, sequence_length)
    X_test_gru, y_test_gru = create_sequences(test_data, sequence_length)

    # Reshape for GRU input [samples, time steps, features]
    X_train_gru = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_gru = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    ''')

    st.code('''
    # Define the GRU model
    model = Sequential()

    # GRU layer
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_train_gru.shape[1], 1)))
    model.add(Dropout(0.2))

    # Second GRU layer
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Model summary
    model.summary()
            
    ''')

    st.markdown('''
    | Layer (type)        | Output Shape | Param # |
    |---------------------|--------------|---------|
    | gru_4 (GRU)         | (None, 19, 50) | 7,950   |
    | dropout_4 (Dropout) | (None, 19, 50) | 0       |
    | gru_5 (GRU)         | (None, 50)    | 15,300  |
    | dropout_5 (Dropout) | (None, 50)    | 0       |
    | dense_2 (Dense)     | (None, 1)     | 51      |
    ''')
    st.write('Total params: 23,301 (91.02 KB)')
    st.write('Trainable params: 23,301 (91.02 KB)')
    st.write('Non-trainable params: 0 (0.00 B)')

    st.code('''
    # Train the model
    history = model.fit(X_train_gru[:len(y_train_gru)], 
                        y_train_gru, 
                        epochs=20, 
                        batch_size=32, 
                        validation_data=(X_test_gru[:len(y_test_gru)], 
                        y_test_gru))
    ''')

    st.code('''
    # Predict on test data
    predicted_prices = model.predict(X_test_gru)

    # Inverse transform to get actual price predictions
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

    # Rescale y_test back to original scale
    y_test_actual = scaler.inverse_transform(y_test_gru.reshape(-1, 1))

    # Ensure both arrays have the same length
    predicted_prices = predicted_prices[:len(y_test_actual)]

    # Evaluation
    mae = mean_absolute_error(y_test_actual, predicted_prices)
    rmse = root_mean_squared_error(y_test_actual, predicted_prices)
    mse = np.square(rmse)
    r2 = r2_score(y_test_actual, predicted_prices)

    # append GRU results
    model_name = 'GRU'
    if model_name in evaluation_df['Model'].values:
        evaluation_df.loc[evaluation_df['Model'] == model_name, ['MAE', 'MSE', 'RMSE', 'R2']] = [mae, mse, rmse, r2]
    else:
        evaluation_df.loc[len(evaluation_df)] = [model_name, mae, mse, rmse, r2]
    evaluation_df
    ''')

    url14 = "https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/main/data/evaluation_df8.csv"
    @st.cache_data
    def load_data():
        return pd.read_csv(url14)
    df14 = load_data()
    st.dataframe(df14)

    st.markdown('''
    Plot the predicted vs actual stock prices for this model
    ''')

    st.code('''
    import matplotlib.pyplot as plt

    # Plot the actual vs predicted prices
    plt.figure(figsize=(20,8))
    plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
    plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
    plt.title('GRU Model - Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/GRU.png')

    st.code('''
        # Get feature importances
    GRU_feature_importance = best_lgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': GRU_feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Plot all the feature importances
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', dodge=False, legend=False)
    plt.title('Feature Importance from GRU', fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, linestyle='--', alpha=0.7)
    ''')

    st.image('https://raw.githubusercontent.com/Sami-Alyasin/Crystal-Stockball/refs/heads/main/data/featureimportance6.png')

    st.markdown('''
    Since the LightBGM model performed the best, we will use it to predict the stock price movement.
    ''')

    st.code('''
    import joblib

    # Export the LightGBM model
    joblib.dump(lgb_model, 'lgb_model.pkl')
            
    ''')
    
    st.markdown('''
    ### The code below is what we will be deploying for the live model in the app
    It combines all the steps we need to predict the stock price movement using the LightBGM model
     - Collect the data, create additional features, load the trained model, and predict the stock price movement based on the user's selection
    ''')
    
    st.code('''
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

st.set_page_config(layout="centered",initial_sidebar_state="expanded")

# Load the saved LightGBM model
lgb_model = joblib.load('lgb_model.pkl')

# Load sector data
sector_data = pd.read_csv('sector_data.csv')

# Streamlit layout
st.markdown("<h1 style='text-align: center;'>The Crystal Stockball: Fortune Telling for Investors</h1>", unsafe_allow_html=True)

st.divider()
left, middle, right = st.columns(3,gap='large')  
with left:
    st.page_link("Pages/Home.py",label = "About", icon = "📝")
    
with middle:
    st.page_link("Pages/Project.py",label = "Project walkthorugh", icon = "📚")  
    
with right:
    st.page_link("Pages/LiveModel.py",label = "Give the model a try!", icon = "🔮")
    
    
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

# df.to_csv('stock_data.csv', index=False)

def predict_tomorrow_stock_price(ticker):
    
    # Filter for the desired stock
    stock_data = df[df['Ticker'] == ticker]
    
    # Get the latest feature values
    last_row = stock_data.iloc[-1]
    
    # Prepare the next day's features
    X_tomorrow = pd.DataFrame({
    'Daily_Return': last_row['Daily_Return'],  
    'Volatility': stock_data['Daily_Return'].rolling(window=30).std().iloc[-1],
    'Z_Score_ACP': last_row['Z_Score_ACP'],
    'Z_Score_Volume': last_row['Z_Score_Volume'],
    'RM_30': stock_data['Adj Close'].rolling(window=30).mean().iloc[-1],
    'RSTD_30': stock_data['Adj Close'].rolling(window=30).std().iloc[-1],
    'BB_upper_30': stock_data['Adj Close'].rolling(window=30).mean().iloc[-1] + (2 * stock_data['Adj Close'].rolling(window=30).std().iloc[-1]),
    'BB_lower_30': stock_data['Adj Close'].rolling(window=30).mean().iloc[-1] - (2 * stock_data['Adj Close'].rolling(window=30).std().iloc[-1]),
    'RM_60': stock_data['Adj Close'].rolling(window=60).mean().iloc[-1],
    'RSTD_60': stock_data['Adj Close'].rolling(window=60).std().iloc[-1],
    'BB_upper_60': stock_data['Adj Close'].rolling(window=60).mean().iloc[-1] + (2 * stock_data['Adj Close'].rolling(window=60).std().iloc[-1]),
    'BB_lower_60': stock_data['Adj Close'].rolling(window=60).mean().iloc[-1] - (2 * stock_data['Adj Close'].rolling(window=60).std().iloc[-1]),
    'RM_90': stock_data['Adj Close'].rolling(window=90).mean().iloc[-1],
    'RSTD_90': stock_data['Adj Close'].rolling(window=90).std().iloc[-1],
    'BB_upper_90': stock_data['Adj Close'].rolling(window=90).mean().iloc[-1] + (2 * stock_data['Adj Close'].rolling(window=90).std().iloc[-1]),
    'BB_lower_90': stock_data['Adj Close'].rolling(window=90).mean().iloc[-1] - (2 * stock_data['Adj Close'].rolling(window=90).std().iloc[-1]),
    'RSI_30': calculate_rsi(stock_data['Adj Close'], window=30).iloc[-1],
    'RSI_60': calculate_rsi(stock_data['Adj Close'], window=60).iloc[-1],
    'RSI_90': calculate_rsi(stock_data['Adj Close'], window=90).iloc[-1]
    }, index=[0])   

    # Predict tomorrow's price
    tomorrow_prediction = lgb_model.predict(X_tomorrow)
    return tomorrow_prediction[0]


if st.button('Predict Tomorrow\'s Price'):
    tomorrow_price = predict_tomorrow_stock_price(selected_ticker)
    st.markdown(f"<h2 style='text-align: center;'>🎯 Predicted adjusted close price for {selected_ticker} is: ${tomorrow_price:.2f}</h2>", unsafe_allow_html=True)
    # print the last known adjusted closing price before the prediction
    st.write(f"Last known adjusted closing price for {selected_ticker} was ${df[df['Ticker'] == selected_ticker]['Adj Close'].iloc[-1]:.2f}")         
''')