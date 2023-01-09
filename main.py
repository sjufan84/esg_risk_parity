
#Import modules
from alpaca_trade_api import TimeFrame
import pandas as pd
import requests
import numpy as np
import json
from dotenv import load_dotenv
import sys
import os
from pathlib import Path
import datetime as dt

import matplotlib.pyplot as plt
import plotly.express as px
import holoviews as hv
import matplotlib.pyplot as plt
import streamlit as st



from utils.MCForecastTools import MCSimulation
from utils.AlpacaFunctions import get_historical_dataframe, get_crypto_bars
from utils.get_crypto_model import getCryptoModel
from utils.get_stock_model import getStockModel

import warnings
warnings.filterwarnings('ignore')



# %% [markdown]
# #### We first pull data for ETHUSD and BTCUSD to extract close data from 1095 calendar days to today using Alpaca's Crypto API

# %%
# Constructing crypto api call
# For the purposes of demonstration through Voila we are removing the input options
numDays = 1095
# Create a yes/no input to determine if user would like to add crypto to their portfolio
crypto_input = st.checkbox('Would you like to add crypto to your portfolio?')
if crypto_input:
    symbol = 



#if crypto_input == 'y':
#   symbol = input('List which coin you would like to add, i.e. BTCUSD, ETHUSD')
crypto_tickers = ['BTCUSD', 'ETHUSD']
today = dt.date.today()
start = (today - dt.timedelta(days=numDays)).isoformat()
yesterday = (today - dt.timedelta(days=1)).isoformat()
end = yesterday
timeframe='1Day'
limit = 5000
crypto_df = pd.DataFrame()
crypto_close_df = pd.DataFrame()

# Iterating through tickers to isolate and concat close data 

for symbol in crypto_tickers:
    crypto_df = get_crypto_bars(symbol=symbol, start=start, end=end, timeframe=timeframe, limit=limit)
    crypto_close = pd.DataFrame(crypto_df['close'])
    crypto_close.columns = [symbol]
    crypto_close_df = pd.concat([crypto_close, crypto_close_df], axis=1)
    crypto_close_df = crypto_close_df[~crypto_close_df.index.duplicated(keep='first')]
    print(crypto_close_df)
    

# %%
# Here we read in our esg.csv file for combined stocks from the S and P 500 as well as the Nasdaq 100
esg_combined = pd.read_csv(Path('./esg/combined_esg.csv'), index_col = [0])

# %%
# Establish "ticker" column in our dataframe using index
esg_combined['ticker'] = list(esg_combined.index)

# %% [markdown]
# #### After reading in our .csv file with ESG data for the S and P 500 universe we isolate the top 3 stocks from each category to include in our portfolio model

# %%
# For the purposes of demonstration in Voila we are only using the top 3 rated env stocks
# and adding them to our portfolio as well as plotting the scores and tickers
env_input = 3
env_top = esg_combined.sort_values(by='environment_score').iloc[-env_input:, :]
env_plot = env_top.hvplot.bar(y='environment_score', color='green', hover_cols = 'Name', rot=90, title = 'Top 3 Rated Environment Stocks')
display(env_plot)
env_tickers = list(env_top.index)

# %%
# For the purposes of demonstration in Voila we are only using the top 3 rated social stocks
# and adding them to our portfolio as well as plotting the scores and tickers
social_input = 3
social_top = esg_combined.sort_values(by='social_score').iloc[-social_input:, :]
social_plot = social_top.hvplot.bar(y='social_score', color='blue', hover_cols = 'Name', rot=90, title = 'Top 3 Rated Social Stocks')
display(social_plot)
social_tickers = list(social_top.index)

# %%
# For the purposes of demonstration in Voila we are only using the top 3 rated governance stocks
# and adding them to our portfolio as well as plotting the scores and tickers
gov_input = 3
gov_top = esg_combined.sort_values(by='governance_score').iloc[-gov_input:, :]
gov_plot = gov_top.hvplot.bar(y='governance_score', color='red', hover_cols = 'Name', rot=90, title = 'Top 3 Rated Governance Stocks')
display(gov_plot)
gov_tickers = list(gov_top.index)

# %% [markdown]
# #### After extracting and plotting the top 3 stocks from each category, we construct our portfolio by adding these 9 stocks to our crypto portfolio and then get historical stock data from Alpaca for further calculations

# %%
# Concatentate social, gov, and env tickers
stock_tickers=env_tickers + gov_tickers + social_tickers

# Add crypto tickers to portfolio if applicable
portfolio_tickers = stock_tickers + crypto_tickers
 
# Display portfolio tickers
portfolio_status = f'Your portfolio now includes: {portfolio_tickers}'
print(portfolio_status)

# %%
# Constructing stock data api call
symbols = stock_tickers
numDay = numDays
today = dt.date.today()
start = (today - dt.timedelta(days=numDays)).isoformat()
yesterday = (today - dt.timedelta(days=1)).isoformat()
end= yesterday
timeframe='1Day'
limit = 5000
stocks_df = pd.DataFrame()
stocks_close_df = pd.DataFrame()

# Iterating through tickers to isolate and concat close data 
for symbol in symbols:    
    try:
        symbol_df = get_historical_dataframe(symbol=symbol, start=start, end=end, timeframe=timeframe)
        ticker_close_df = pd.DataFrame(symbol_df['close'])
        ticker_close_df.columns = [symbol]
        stocks_close_df = pd.concat([stocks_close_df, ticker_close_df], axis=1)
        stocks_close_df = stocks_close_df[~stocks_close_df.index.duplicated(keep='first')]
        stocks_df = pd.concat([stocks_df, symbol_df])
    except:
        print(symbol)
        pass
display(stocks_close_df)
    

# %%
#Drop n/a values by columns, we don't want to skew our data if stocks do not have enough historical data
stocks_df.dropna(axis=1, inplace=True)

# %%
#Eliminating any duplicate columns
new_stocks_df = stocks_df.copy().loc[:,~stocks_df.columns.duplicated()]

# %%
# Normalizing indices to concat dataframes
crypto_close_df.index = crypto_close_df.index.normalize()
stocks_close_df.index = stocks_close_df.index.normalize()
portfolio_close_df = pd.DataFrame()
portfolio_close_df = pd.concat([crypto_close_df, stocks_close_df], axis = 1).dropna()

# %% [markdown]
# #### Below we generate a new dataframe with pct change calculations for our portfolio assets to be used for our risk parity calcuations

# %%
# Establishing 'Y' for risk parity calculations which will be the daily returns of our portfolio
Y = portfolio_close_df.pct_change().dropna()
Y

# %%
mc_df = portfolio_close_df.copy()

# %% [markdown]
# #### In order to format our data for Monte Carlo simulations, we need to create a multi-index dataframe with closing prices

# %%
# We need to convert our dataframe into a multiindex for MC simulations to run correctly
close_list = ['close'] * mc_df.shape[1]
close_list = list(close_list)
mc_df.columns = pd.MultiIndex.from_arrays([list(mc_df.columns), close_list])
mc_df

# %% [markdown]
# #### Below we are building our portfolio weights using our HRP model with pre-set values for risk, objectives, etc.

# %%
# Building the portfolio object
import riskfolio as rp

port = rp.HCPortfolio(returns=Y)

# Estimate optimal portfolio:

model='HRP' # Could be HRP or HERC
codependence = 'pearson' # Correlation matrix used to group assets in clusters
rm = 'MDD' # Risk measure used, this time will be variance
rf = 0 # Risk free rate
linkage = 'single' # Linkage method used to build clusters
max_k = 10 # Max number of clusters used in two difference gap statistic, only for HERC model
leaf_order = True # Consider optimal order of leafs in dendrogram
obj = 'Utility'
l=2


w = port.optimization(model=model,
                      codependence=codependence,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      max_k=max_k,
                      leaf_order=leaf_order,
                      obj = obj,
                      l=l)

display(w.T)

# %%
weights_df = pd.DataFrame(w.round(4))

# %% [markdown]
# #### After generating our portfolio weights we plot them using a pie chart as well as bar for visualization purposes

# %%
# Plotting the composition of the portfolio

ax = px.pie(weights_df, title='Portfolio composition', names = list(weights_df.index), values = 'weights')
ax

# %%
ax = px.bar(weights_df.sort_values(by='weights'), title='Portfolio composition', y='weights')
ax

# %%
# Formatting weights for use in generating model returns
weights = w.iloc[:, 0]

# %% [markdown]
# #### For demonstration purposes we will hard code $100000 as our portfolio total and then break up the amount by weights of assets
# 

# %%
portfolio_total = 100000
weights_df['initial_capital'] = (weights_df['weights'] * portfolio_total).round(3)
weights_df

# %% [markdown]
# #### After establishing our portfolio weights and initial capital for each asset, we call our getStocks and getCrypto models to predict buy / sell opportunities and then compare the results to the actual portfolio returns -- this will generate a plot for each assets model vs. actual returns in new windows for visualization

# %%
# Here we call our getStockModel script which uses our machine learning models to predict
# buy and sell orders for each stock
trades_df_dict = {}
new_tickers = []
actual_cumprod = []
for ticker in stock_tickers:
    if weights_df.loc[ticker]['weights'] != 0.0:
        try:
            initial_capital =  weights_df.loc[ticker]['initial_capital']
            ticker_trades_df = getStockModel(ticker = ticker, numDays = numDays, initial_capital = initial_capital)
            cumprod = (1 + Y[ticker].loc[ticker_trades_df.index.normalize()]).cumprod()
            trades_df_dict[ticker] = ticker_trades_df
            new_tickers.append(ticker)
            actual_cumprod.append(cumprod)
        except Exception:
            print(ticker)
            pass

# %%
# Getting dataframe for crypto ticker returns and concatenating to stocks df
# using our crypto model with different machine learning techniques
# then generating the ending cash balance for each asset
crypto_cash_list = []
for ticker in crypto_tickers:
    #if weights_df.loc[ticker]['weights'] != 0.0:
    initial_capital =  weights_df.loc[ticker]['initial_capital']
    crypto_trades_df = getCryptoModel(ticker = ticker, numDays = numDays, initial_capital = initial_capital)
    cumprod = (1 + Y[ticker].loc[ticker_trades_df.index.normalize()]).cumprod()
    trades_df_dict[ticker] = crypto_trades_df
    #new_tickers.append(ticker)
    actual_cumprod.append(cumprod)
    crypto_cash = crypto_trades_df['Portfolio Cash'][-1]
    crypto_cash_list.append(crypto_cash)
        


# %%
# Calculating cash on hand from model returns
model_cash_list = []
for ticker in new_tickers:
    try:
        ticker_cash = trades_df_dict[ticker]['Portfolio Cash'][-1]
        model_cash_list.append(ticker_cash)
    except:
        print(ticker)

# Adding crypto cash balances to our model cash list
model_cash_list += crypto_cash_list
# Adding crypto tickers to our new ticker list
new_tickers += crypto_tickers 

# %% [markdown]
# #### Once we have run our models for our portfolio, we will then pull data to compare the total returns for our model vs. actual and present it to the user

# %%
# Converting our model_cash_list to a series for concatenation
model_cash_series = pd.Series(model_cash_list, index = new_tickers)
model_cash_series.columns = ['model_cash']

# %%
# Creating df with initial capital amounts, ending cash balance and weights for assets
# in our portfolio
model_cash_df = pd.concat([weights_df, model_cash_series], axis=1).dropna()

# %% [markdown]
# #### Presenting a dataframe to examine our ending model cash positions vs. initial capital

# %%
# Comparing model returns with initial capital invested
model_cash_df.rename(columns = {0 : 'model_cash'}, inplace=True)
model_cash_df

# %%
# Calculating model profit
portfolio_cash = model_cash_series.sum().round(2)
initial_capital = weights_df['initial_capital'].sum().round(2) 
profit = portfolio_cash - initial_capital
profit_percent = ((profit / initial_capital) * 100).round(2)
model_profit_statement = f'Your model profit for this period would have been {profit} or {profit_percent}%'


# %%
# Calculating what actual returns would have been
actual_cumprod_df = pd.DataFrame(actual_cumprod, index = new_tickers)
actual_cumprod_df = pd.DataFrame((actual_cumprod_df.iloc[:, -1]).round(2))
actual_cumprod_df.columns = ['actual_returns']

# %%
actual_returns_df = pd.concat([weights_df, actual_cumprod_df], axis=1).dropna()

# %%
# Concatenating actual and model returns for comparison
actual_returns_df['final_cash'] = (actual_returns_df['initial_capital'] * actual_returns_df['actual_returns']).round(2)
actual_returns_df = pd.concat([actual_returns_df, model_cash_df], axis=1).dropna()
actual_returns_df

# %%
# Comparing real and model returns and displaying to user
actual_profit = actual_returns_df['final_cash'].sum().round(2) - initial_capital
percent_profit = (actual_profit / initial_capital) * 100
actual_profit_statement = f'The real returns for the portfolio would have been {actual_profit} or {percent_profit}%'
display(model_profit_statement, actual_profit_statement)

# %% [markdown]
# #### We will now move on to running 100 Monte Carlo simulations to model theoretical returns for 252 trading days using our historical data as well as the weights we have established prior.

# %%
# Running Monte Carlo Simulations based on calculated portfolio
for ticker in new_tickers:
    trades_df_dict[ticker].columns = pd.MultiIndex.from_product([[ticker], np.array(trades_df_dict[ticker].columns)])

# %%
# Establishing asset weights for our MC Simulations
weights = weights_df['weights'].values

# %%
# Run the simulations:
portfolio_sim = MCSimulation(mc_df, weights = weights, num_simulation=100, num_trading_days=252)
#arkk_sim = MCSimulation(arkk, num_simulation = 200, num_trading_days = 504)
#qqq_sim = MCSimulation(qqq, num_simulation = 200, num_trading_days = 504)
#spy_sim = MCSimulation(spy, num_simulation = 200, num_trading_days = 504)
sim_returns = portfolio_sim.calc_cumulative_return()
#arkk_returns = arkk_sim.calc_cumulative_return()
#qqq_returns = qqq_sim.calc_cumulative_return()
#spy_returns = spy_sim.calc_cumulative_return()

# %% [markdown]
# #### After we have run the simulations we will present a plot of the median returns as well as the distribution of the resulting data

# %%
# Display median returns of Monte Carlo Simulation
returns_median = sim_returns
for i in range(0, len(sim_returns.index) - 1, 1):
    returns_median.iloc[i] = sim_returns.iloc[i].median()
returns_median = pd.DataFrame(returns_median.loc[:, 0])
returns_median.columns = ['median_return']
returns_median.index.name = 'days'
returns_plot = returns_median.hvplot(label = 'Median Portfolio Returns')
returns_plot

# %%
# Summary of cumulative return simulations
portfolio_sim.summarize_cumulative_return()

# %%



