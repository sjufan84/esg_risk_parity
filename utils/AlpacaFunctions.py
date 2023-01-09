import datetime as dt
from dotenv import load_dotenv
import sys
import os
import pandas as pd
from alpaca.data import Trade, Snapshot, Quote, Bar
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
    StockLatestTradeRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
    StockLatestBarRequest,
)
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoQuotesRequest,
    CryptoTradesRequest,
    CryptoLatestTradeRequest,
    CryptoLatestQuoteRequest,
    CryptoSnapshotRequest,
    CryptoLatestBarRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Exchange, DataFeed
from alpaca.data.models import BarSet, QuoteSet, TradeSet


#import environment variables
load_dotenv()
APCA_API_KEY_ID = os.getenv('APCA-API-KEY-ID')
APCA_API_SECRET_KEY = os.getenv('APCA-SECRET-KEY')

# no keys required.
crypto_client = CryptoHistoricalDataClient()

# keys required
stock_client = StockHistoricalDataClient(APCA_API_KEY_ID,  APCA_API_SECRET_KEY)       


# Get bars:
def get_historical_dataframe(symbol, timeframe, start, end, limit):
    request = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end = end, limit=limit
    )
    barset = stock_client.get_stock_bars(request_params=request)
    stocks_df = barset.df
    return stocks_df

    
# Get historical Crypto data:
def get_crypto_bars(symbol, timeframe, start, end, limit):
    crypto_request_params = CryptoBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Day,
                        start=start,
                        end=end)
    bars = crypto_client.get_crypto_bars(crypto_request_params)
    # convert to dataframe
    bars.df
    crypto_df = bars.df    
    return crypto_df

#def get_news(symbol, start, end, limit):
#    news_df = alpaca.get_news(symbol=symbol, start=start, end=end, limit=limit)
#    return news_df


