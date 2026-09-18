#First things first: import the necessary packages
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#Maximize dataframe display options:
ticker_df = pd.set_option("display.max_columns", 100)
ticker_df = pd.set_option("display.max_rows", 100)
ticker_df = pd.set_option("display.width", 100)


#Defining the periods to fetch the ticker and it's market data
start_date = datetime(year=2025, month=1, day=1)
end_date = datetime.today()


#Download market data for out ticker
ticker_df = yf.download(
    "ETH-USD", start=start_date, end=end_date, interval="1d", auto_adjust=True, rounding=False, progress=True
    )

#Moving each date and price one up, so yesterdays price is now in today's price before yesterday's price is in yesterday as if we deleted a day and pushed the data one up
ticker_df = ticker_df.shift(periods=1)
#But because we did that step of pushing the data up by one day we're left with one cell the first that's empty why because we pushed it so the data can't fill itself but that's fine we can clean it the following way
ticker_df = ticker_df.dropna()
#Turn rounded prices into thousands:
ticker_df = np.round(ticker_df, 2)
#Removing redundant data as soon as we can
ticker_df = ticker_df.drop("Open", axis=1)

#HOMEWORK:
# LAG FEATURES:
# Add column for price_yesterday, price_2_days_ago and so on (try to do a week) so the model can have a historical timeline to look at

ticker_df.insert(loc=0, column="Price 1 day_ago", value=ticker_df["Close"].shift(1))
ticker_df.insert(loc=1, column="Price 2 days ago", value=ticker_df["Close"].shift(2))
ticker_df.insert(loc=2, column="Price 3 days ago", value=ticker_df["Close"].shift(3))
ticker_df.insert(loc=3, column="Price 4 days ago", value=ticker_df["Close"].shift(4))
ticker_df.insert(loc=4, column="Price 5 days ago", value=ticker_df["Close"].shift(5))
ticker_df.insert(loc=5, column="Price 6 days ago", value=ticker_df["Close"].shift(6))
ticker_df.insert(loc=6, column="Price 7 days ago", value=ticker_df["Close"].shift(7))
ticker_df.insert(loc=7, column="Price Tommorow", value=ticker_df["Close"].shift(-1))

#print(ticker_df)
# MOVING AVERAGES:
# Create a 7 days and 30 days and 90 days moving average -> Actually allows the model to be very precise knowing where the trend is going
eth_close = ticker_df["Close"]
#week sma
periods = 7
eth_7_sma = np.round(eth_close.rolling(window=periods).mean(), 2)
ticker_df.insert(loc=7, column="7 SMA", value=eth_7_sma)

#month sma:
periods = 30
eth_month_sma = np.round(eth_close.rolling(window=periods).mean(), 2)
ticker_df.insert(loc=8, column="30 SMA", value=eth_month_sma)

#90 days sma

periods = 90
ninety_sma = np.round(eth_close.rolling(window=periods).mean(), 2)
ticker_df.insert(loc=9, column="90 SMA", value=ninety_sma)

#Since price one day ago is just yesterdays close and price 2 days ago is before yesterdays close it just acs as today which we want to predict we drop it:
ticker_df = ticker_df.sort_index(axis=1).drop("Close", axis=1)
#drop dead cells:
ticker_df = ticker_df.dropna()
print(ticker_df)
# DAILY RETURNS:
#Create a column showing the change percentage from the previous day -> Allowing the model to learn day-to-day volatility of the asset
# TIME INDICATORS:
#Break the date column into Month and day_of__week -> sopme asets trade differently throughout the week