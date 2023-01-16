
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

from utils.risk_parity import get_parity_model
from utils.get_stock_data import get_stock_data


from utils.MCForecastTools import MCSimulation
from utils.AlpacaFunctions import get_crypto_bars
from utils.get_crypto_model import getCryptoModel
from utils.get_stock_model import getStockModel

import warnings
warnings.filterwarnings('ignore')

if 'Y' not in st.session_state:
    st.session_state['Y'] = pd.DataFrame()
if 'w' not in st.session_state:
    st.session_state['w'] = pd.DataFrame()
if 'rm' not in st.session_state:
    st.session_state['rm'] = 'equal'


# %% [markdown]
# #### We first pull data for ETHUSD and BTCUSD to extract close data from 1095 calendar days to today using Alpaca's Crypto API

# %%
# Constructing crypto api call

crypto_input = st.radio('Would you like to add crypto to your portfolio?', ('y', 'n'))
# If user selects yes, create a multiselect input to give them the options of "ETH/USD" and "BTC/USD"
if crypto_input == 'y':
    crypto_tickers = st.multiselect('Select which crypto you would like to add to your portfolio', ('ETH/USD', 'BTC/USD'))
    numDays = st.number_input('How many days of data would you like to use for your portfolio?', min_value=1, max_value=20000, value=1500, step=1)

# If user selects no, set crypto_tickers to an empty list
else:
    crypto_tickers = []
if crypto_input == 'y':
    crypto_submit = st.button('Retrieve data for crypto')
    if crypto_submit:

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

elif crypto_input == 'n':
    crypto_close_df = pd.DataFrame()
        

# %%
# Here we read in our esg.csv file for combined stocks from the S and P 500 as well as the Nasdaq 100
esg_combined = pd.read_csv(Path('./esg/combined_esg.csv'), index_col = [0])

# %%
# Establish "ticker" column in our dataframe using index
esg_combined['ticker'] = list(esg_combined.index)

# %% [markdown]
# After reading in our .csv file with ESG data for the S and P 500 universe we isolate the top 3 stocks from each category to include in our portfolio model

# %%
# For the purposes of demonstration in Voila we are only using the top 3 rated social stocks
# and adding them to our portfolio as well as plotting the scores and tickers
st.markdown("#### Select up to 3 top rated Environment stocks to add to your portfolio")
top_env_stocks = esg_combined.sort_values(by='environment_score').iloc[-20:, :]
# Allow the user to select up to 3 top rated env stocks to add to their portfolio
env_input = st.multiselect('Select up to 3 stocks to add to your portfolio', top_env_stocks['Name'])

# Create an hvplot bar chart of the user selected stocks
env_top = top_env_stocks[top_env_stocks['Name'].isin(env_input)]
# Create a bar chart of the user selected stocks that highlights the selected stocks from the top_env_stocks dataframe
fig = px.bar(top_env_stocks, x='Name', y='environment_score', color='Name', color_discrete_sequence=['green'])  
# Add a vertical line to the bar chart to highlight the selected stocks
for env in env_input:
    fig.add_vline(x=env, line_width=3, line_dash="dash", line_color="red")

fig.update_layout(title_text='Top 3 Rated Env Stocks')
st.plotly_chart(fig)


# env_plot = env_top.hvplot.bar(y='environment_score', color='red', hover_cols = 'Name', rot=90, title = 'Top 3 Rated Env Stocks')
# st.bokeh_chart(hv.render(env_plot))
env_tickers = list(env_top.index)

## %%
# For the purposes of demonstration in Voila we are only using the top 3 rated social stocks
# and adding them to our portfolio as well as plotting the scores and tickers
st.markdown("#### Select up to 3 top rated socialernance stocks to add to your portfolio")
top_social_stocks = esg_combined.sort_values(by='social_score').iloc[-20:, :]
# Allow the user to select up to 3 top rated social stocks to add to their portfolio
social_input = st.multiselect('Select up to 3 stocks to add to your portfolio', top_social_stocks['Name'])

# Create a plotly bar chart of the user selected stocks
social_top = top_social_stocks[top_social_stocks['Name'].isin(social_input)]
# Create a bar chart of the user selected stocks that highlights the selected stocks from the top_social_stocks dataframe
fig = px.bar(top_social_stocks, x='Name', y='social_score', color='Name', color_discrete_sequence=['blue'])  
# Add a vertical line to the bar chart to highlight the selected stocks
for social in social_input:
    fig.add_vline(x=social, line_width=3, line_dash="dash", line_color="red")

fig.update_layout(title_text='Top 3 Rated Env Stocks')
st.plotly_chart(fig)
social_tickers = list(social_top.index)

# %%
# For the purposes of demonstration in Voila we are only using the top 3 rated governance stocks
# and adding them to our portfolio as well as plotting the scores and tickers
st.markdown("#### Select up to 3 top rated governance stocks to add to your portfolio")
top_gov_stocks = esg_combined.sort_values(by='governance_score').iloc[-20:, :]
# Allow the user to select up to 3 top rated governance stocks to add to their portfolio
gov_input = st.multiselect('Select up to 3 stocks to add to your portfolio', top_gov_stocks['Name'])

# Create a plotly bar chart of the user selected stocks
gov_top = top_gov_stocks[top_gov_stocks['Name'].isin(gov_input)]
# Create a bar chart of the user selected stocks that highlights the selected stocks from the top_gov_stocks dataframe
fig = px.bar(top_gov_stocks, x='Name', y='governance_score', color='Name', color_discrete_sequence=['yellow'])  
# Add a vertical line to the bar chart to highlight the selected stocks
for gov in gov_input:
    fig.add_vline(x=gov, line_width=3, line_dash="dash", line_color="red")

fig.update_layout(title_text='Top 3 Rated Env Stocks')
st.plotly_chart(fig)
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
st.markdown(portfolio_status)
stocks_close_df = pd.DataFrame()
Y = pd.DataFrame()


with st.form('get_stock_data', clear_on_submit = True):
    numDays = st.number_input('How many days of historical data would you like to retrieve?', min_value=1, max_value=1000, value=1000, step=1)
    submit_stock_data = st.form_submit_button('Retrieve Stock Data')
# Retrieve stocks_close_df, new_stocks_df, portfolio_close_df from get_stock_data function
    if submit_stock_data: 
        stocks_close_df = get_stock_data(portfolio_tickers, numDays)
        st.session_state.Y = stocks_close_df.pct_change().dropna()


with st.form('risk_model_form', clear_on_submit=True):
    risk_measures = ['MV', 'KT', 'MAD', 'MSV', 'SKT', 'FLPM', 'SLPM', 'VaR', 'CVaR', 'TG', 'EVaR', 'WR', 'RG', 'CVRG', 'TGRG', 'MDD', 'ADD', 'DaR', 'CDaR', 'EDaR', 'UCI', 'MDD_Rel', 'ADD_Rel', 'DaR_Rel', 'CDaR_Rel', 'EDaR_Rel', 'UCI_Rel']
    risk_selection = st.selectbox('Select a risk measure', risk_measures, index=0, key='risk_selection')
    if risk_selection != '':
        st.session_state.rm = risk_selection
    parity_button = st.form_submit_button('Create Risk Model')
    if parity_button:
        # Create a risk parity model
        st.session_state.w = get_parity_model(st.session_state.Y, st.session_state.rm)
        # Display risk parity model
        



# %% [markdown]
# #### For demonstration purposes we will hard code $100000 as our portfolio total and then break up the amount by weights of assets
# 

# %%
# Asking the user how much they would like to invest
'''
# %%
    # Formatting weights for use in generating model returns
    weights = st.session_state.w.iloc[:, 0]

portfolio_total = st.number_input('How much would you like to invest in your portfolio?', min_value=1000, max_value=100000000, value=100000)
portfolio_total_sumbit = st.form_submit_button('Submit')
if portfolio_total_sumbit:
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
    st.write(model_profit_statement, actual_profit_statement)

    # %% [markdown]
    # #### We will now move on to running 100 Monte Carlo simulations to model theoretical returns for 252 trading days using our historical data as well as the weights we have established prior.
'''
    # %%
    