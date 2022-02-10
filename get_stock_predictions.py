# %%
#imports
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import hvplot.pandas


from pathlib import Path
import datetime as dt

# %%
# import functions
from utils.AlpacaFunctions import get_historical_dataframe, get_crypto_bars, get_news
from utils.data_process import return_rolling_averages
from utils.data_process import return_crossovers
from utils.data_process import return_weighted_crossovers
from utils.neural_functions import shallow_neural
from utils.neural_functions import deep_neural
from utils.eval_functions import sma_crossover_eval
from utils.eval_functions import results_trade_amount_nostop
from utils.eval_functions import results_trade_amount_stops
from utils.eval_functions import buy_or_sell_all_if_available
from utils.eval_functions import buy_or_sell_trade_percent
from utils.sentiment_functions import concat_sentiment
from sentiment_analysis.sentiment_functions import get_sentiments, getTwitterData


def getStockNewsSentiment(symbol, numDays, limit):
    # Get news for ticker
    today = dt.date.today()
    two_years_ago = (today - dt.timedelta(days=numDays))
    yesterday = (today - dt.timedelta(days=1)).isoformat()
    start = two_years_ago
    end= yesterday
    symbol = symbol
    limit = limit
    news_df = get_news(symbol, start, end, limit)

    # %%
    stories_df = pd.DataFrame(columns = ['date', 'summary'])
    for i in range(0, len(news_df), 1): 

        stories_df.loc[i,'date'] = news_df[i].created_at
        stories_df.loc[i,'summary'] = news_df[i].summary
                        


    # %%
    sentiment_df = get_sentiments(stories_df)
    sentiment_df

    # %%
    sentiment_df.index = stories_df['date']
    sentiment_df

    # %%
    sentiment_df.index = sentiment_df.index.normalize()
    sentiment_df_resampled = sentiment_df.resample('D', convention ='end').mean()
    sentiment_df_resampled

    # %%
    sentiment_df_resampled.fillna(0, inplace=True)
    return sentiment_df_resampled

    
def getPredictions(stock_df):
    close_df = pd.DataFrame(stock_df["close"])
    close_df.index = close_df.index.normalize()

    # %%
    return_rolling_averages(close_df)
    

    # %%
    cross_df = return_crossovers(close_df)

    # %%
    cross_signals=cross_df.sum(axis=1)
    

    # %%
    pct_change_df = close_df.pct_change()
    

    # %%
    cross_weighted_df = return_weighted_crossovers(close_df, pct_change_df)

    # %%
    cross_signals_weighted = pd.DataFrame(cross_weighted_df.sum(axis=1))

    # %%
    volume_df = stock_df[['volume']]
    volume_df.index = volume_df.index.normalize()

    # %%
    signals_input_df = pd.DataFrame()
    signals_input_df = pd.concat([pct_change_df, cross_df, volume_df, pct_change_df, cross_signals, cross_signals_weighted, cross_weighted_df], axis=1)

    # %%
    signals_input_df.dropna(inplace=True)
    signals_input_df = pd.concat([signals_input_df, sentiment_df_resampled], axis=1).fillna(0)
    

    # %%
    from finta import TA
    finta_df = pd.DataFrame()

    finta_df['vama'] = TA.VAMA(stock_df)
    finta_df['rsi'] = TA.RSI(stock_df)
    finta_df['ao'] = TA.AO(stock_df)
    finta_df['ema'] = TA.EMA(stock_df)
    finta_df['evwma'] = TA.EVWMA(stock_df)
    finta_df['vfi'] = TA.VFI(stock_df)
    finta_df['msd'] = TA.MSD(stock_df)
    finta_df['efi'] = TA.EFI(stock_df)
    finta_df['stochrsi'] = TA.STOCHRSI(stock_df)
    finta_df['tp'] = TA.TP(stock_df)

    # %%
    finta_df['frama'] = TA.FRAMA(stock_df)
    finta_df['kama'] = TA.KAMA(stock_df)
    finta_df.index = finta_df.index.normalize()

    # %%
    signals_input_df = pd.concat([signals_input_df, finta_df], axis=1)
    

    # %%
    X = signals_input_df.dropna()
    

    # %%
    '''X_resampled = close_df[['close']].loc[X.index]
    X_resampled['close'] = X_resampled.loc[X.index.dayofweek == 0]
    X_resampled.dropna(inplace=True)'''

    # %%
    y_signal = pd.DataFrame(((close_df["close"] > close_df["close"].shift()).shift(-1))*1, index = close_df.index)
    y_signal['close'] = np.where(y_signal['close'] == 0, -1, 1)
    

    # %%
    y_signal = y_signal.loc[X.index]
   

    # %%
    y_signal.iloc[-1]=0
    

    # %%
    train_num = int(numDays * 0.75)
    test_num = numDays-train_num
    # %%
    X_train = X[:train_num]
    X_test = X[-test_num:]


    # %%
    y_train = y_signal[:train_num]
    y_test = y_signal[-test_num:]

    # %%
    scaler = StandardScaler()

    # %%
    X_scaler = scaler.fit(X_train)

    # %%
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = svm.SVC()
    svm_model = svm_model.fit(X_train_scaled, y_train)
    svm_predictions = svm_model.predict(X_train_scaled)
    

    # %%
    shallow_preds = pd.DataFrame(shallow_neural(X_train_scaled, y_train, X_test_scaled, y_test))

    # %%
    deep_preds = pd.DataFrame(deep_neural(X_train_scaled, y_train, X_test_scaled, y_test))

    predictions_df = pd.concat([shallow_preds, deep_preds], axis=1, columns = ['deep_neural', 'svm', 'shallow_neural', 'gbd'])
    

    return predictions_df

    # %%
def backtest(combined_preditions_df): 
    y_shallow['signal'] = shallow_preds
    y_shallow['signal'][0] = 0.0
    y_shallow['signal'] = y_shallow['signal'].apply(lambda x: -1 if x <= -0.25 else x)
    y_shallow['signal'] = y_shallow['signal'].apply(lambda x: 0 if x > -0.25 and x < 0.25 else x)
    y_shallow['signal'] = y_shallow['signal'].apply(lambda x: 1 if x >= 0.25 else x)

    y_shallow['entry/exit'] = y_shallow['signal'].diff().fillna(0)

    y_shallow['close'] = close_df['close'].loc[X_test.index]
    y_shallow.dropna(inplace=True)
    y_shallow


    # %%
    # Set initial capital
    initial_capital = float(100000)

    # Set the share size
    share_size = 600

    y_shallow['position'] = share_size * y_shallow['signal']



    # %%
    y_shallow['entry/exit Position'] = y_shallow['position'].diff().fillna(0)
    y_shallow['Portfolio Holdings'] = y_shallow['close'] * y_shallow['position']


    # %%
    y_shallow['Portfolio Cash'] = initial_capital - (y_shallow['close'] * y_shallow['entry/exit Position']).cumsum()
    y_shallow['Portfolio Total'] = y_shallow['Portfolio Cash'] + y_shallow['Portfolio Holdings']
    y_shallow['Portfolio Daily Returns'] = y_shallow['Portfolio Total'].pct_change()

    y_shallow['Portfolio Cumulative Returns'] = (1 + y_shallow['Portfolio Daily Returns']).cumprod()



    # %%
    pct_change = close_df[['close']].loc[X_test.index]
    pct_change

    # %%
    y_shallow

    # %%
    pct_change = pct_change.pct_change()
    pct_change

    # %%
    pct_change.dropna(inplace=True)

    # %%
    cum_return = (1+pct_change).cumprod() - 1
    cum_return_plot = cum_return.hvplot(label = 'Actual Returns', legend=True)
    shallow_plot = (y_shallow['Portfolio Cumulative Returns'] - 1).hvplot(label = 'Model Returns', legend = True)
    display(cum_return_plot * shallow_plot)
    display(pct_change.std(), y_shallow['Portfolio Daily Returns'].std())
    display(y_shallow['Portfolio Total'].hvplot(yformatter='%.2f'))
    display(y_shallow['Portfolio Cash'].hvplot(yformatter='%.2f'))


    # %%


    # %%
    y_deep = pd.DataFrame()
    y_deep['signal'] = shallow_preds
    y_deep['signal'][0] = 0.0
    y_deep['signal'] = y_deep['signal'].apply(lambda x: -1 if x <= -0.25 else x)
    y_deep['signal'] = y_deep['signal'].apply(lambda x: 0 if x > -0.25 and x < 0.25 else x)
    y_deep['signal'] = y_deep['signal'].apply(lambda x: 1 if x >= 0.25 else x)

    y_deep['entry/exit'] = y_deep['signal'].diff().fillna(0)

    y_deep['close'] = close_df['close'].loc[X_test.index]
    y_deep.dropna(inplace=True)



    # %%
    # Set initial capital
    initial_capital = float(100000)

    # Set the share size
    share_size = 600

    y_deep['position'] = share_size * y_deep['signal']



    # %%


    # %%
    y_deep['entry/exit Position'] = y_deep['position'].diff().fillna(0)
    y_deep['Portfolio Holdings'] = y_deep['close'] * y_deep['position']


    # %%
    y_deep['Portfolio Cash'] = initial_capital - (y_deep['close'] * y_deep['entry/exit Position']).cumsum()
    y_deep['Portfolio Total'] = y_deep['Portfolio Cash'] + y_deep['Portfolio Holdings']
    y_deep['Portfolio Daily Returns'] = y_deep['Portfolio Total'].pct_change()

    y_deep['Portfolio Cumulative Returns'] = (1 + y_deep['Portfolio Daily Returns']).cumprod()



    # %%
    cum_return = (1+pct_change).cumprod() - 1
    cum_return_plot = cum_return.hvplot(label = 'Actual Returns', legend=True)
    deep_plot = (y_deep['Portfolio Cumulative Returns'] - 1).hvplot(label = 'Model Returns', legend = True)
    display(cum_return_plot * shallow_plot)
    display(pct_change.std(), y_deep['Portfolio Daily Returns'].std())
    display(y_deep['Portfolio Total'].hvplot(yformatter='%.2f'))
    display(y_deep['Portfolio Cash'].hvplot(yformatter='%.2f'))


# %%


# %%



