import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from .AlpacaFunctions import get_historical_dataframe


# %%
# Constructing stock data api call
def get_stock_data(stock_tickers, numDays):
    symbols = list(stock_tickers)
    st.write(symbols)
    today = dt.date.today()
    start = pd.to_datetime(today - dt.timedelta(days=numDays))
    yesterday = pd.to_datetime(today - dt.timedelta(days=1))
    end= yesterday
    timeframe='1Day'
    limit = 5000
    stocks_df = pd.DataFrame()
    stocks_close_df = pd.DataFrame()

    
    # Iterating through tickers to isolate and concat close data 
    for symbol in symbols:    
       
        symbol_df = get_historical_dataframe(symbol=symbol, start=start, end=end, timeframe=timeframe, limit = limit)
        ticker_close_df = pd.DataFrame(symbol_df['close'])
        ticker_close_df.index = ticker_close_df.index.droplevel(0)
        ticker_close_df.columns = [symbol]
        if stocks_close_df.empty:
            stocks_close_df = ticker_close_df
        else:
            # Merge the close data into a single dataframe
            stocks_close_df = pd.merge(stocks_close_df, ticker_close_df, left_index=True, right_index=True)
        # Concatenating all stock data
        stocks_df = pd.concat([stocks_df, symbol_df], axis=1)
        
       # %%
    #Drop n/a values by columns, we don't want to skew our data if stocks do not have enough historical data
    stocks_df.dropna(axis=1, inplace=True)

    # %%
    #Eliminating any duplicate columns
    new_stocks_df = stocks_df.copy().loc[:,~stocks_df.columns.duplicated()]
    st.dataframe(stocks_close_df)

    
    
    # Normalize the stock dataframe index if dataframe is not empty
    # Convert index to datetime if not already
    
    stocks_close_df.index = pd.to_datetime(stocks_close_df.index)
    stocks_close_df.index = stocks_close_df.index.normalize()
    
    
    return stocks_close_df