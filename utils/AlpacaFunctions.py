import alpaca_trade_api as tradeapi
import datetime as dt
from dotenv import load_dotenv
import sys
import os
import pandas as pd

#import environment variables
load_dotenv()
APCA_API_KEY_ID = os.getenv('APCA-API-KEY-ID')
APCA_API_SECRET_KEY = os.getenv('APCA-API-SECRET-KEY')


#Create the Alpaca API object -- Is this necessary?
alpaca = tradeapi.REST(APCA_API_KEY_ID, 
                       APCA_API_SECRET_KEY, 
                       api_version='v2')

# Alpaca functions (recently updated from get_barset (deprecated) to get_bars
def get_historical_dataframe(symbol, start, end, timeframe):
    ticker_df = alpaca.get_bars(symbol=symbol, timeframe=timeframe, start=start, end = end, limit = 5000).df
    return ticker_df
def filter_close_prices(dataframe):
    df_close = pd.DataFrame()
    df_close['close'] = dataframe['close']
    return df_close
def calc_daily_returns(df_close_prices):
    daily_returns = df_close_prices.pct_change().dropna()
    return daily_returns

