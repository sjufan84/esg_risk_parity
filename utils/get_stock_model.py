
#imports
import pandas as pd
import numpy as np

# Tensorflow / keras for neural models
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


# Multiple types of models from sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import hvplot.pandas
import selenium
import chromedriver_py
import chromedriver_binary

from finta import TA

from pathlib import Path
import datetime as dt

# %%
# import functions
from utils.AlpacaFunctions import get_historical_dataframe, get_news
from utils.data_process import return_rolling_averages
from utils.data_process import return_crossovers
from utils.data_process import return_weighted_crossovers
from utils.neural_functions import shallow_neural
from utils.neural_functions import deep_neural
from utils.sentiment_functions import get_sentiments


import datetime as dt


# Constructing crypto api call

'''today = dt.date.today()
two_years_ago = (today - dt.timedelta(days=500))
yesterday = (today - dt.timedelta(days=1)).isoformat()
start = two_years_ago
end= yesterday
symbols = ['ETHUSD']
timeframe='1Day'
limit = 5000
stock_df = pd.DataFrame()

# Iterating through tickers to isolate and concat close data
for symbol in symbols:     
    symbol_df = get_crypto_bars(symbol=symbol, timeframe=timeframe, start=start, end=end, limit=limit)
    stock_df = pd.concat([stock_df, symbol_df], axis=1)
    
stock_df'''



# %%
# Construct stock data api call
def get_stock_model(ticker, numDays, initial_capital):

    numDays = numDays
    today = dt.date.today()
    start = today - dt.timedelta(days=numDays)
    yesterday = (today - dt.timedelta(days=1)).isoformat()
    end= yesterday
    symbol = ticker
    timeframe='1Day'
    stock_df = pd.DataFrame()

    stock_df = get_historical_dataframe(symbol=ticker, start=start, end=end, timeframe=timeframe)
    stock_df

    
    # Get news for ticker
    news_df = get_news(symbol=ticker, start=start, end=end, limit=5000)

    # Iterating through returned news objects to create a dataframe that can be used for sentiment analysis

    stories_df = pd.DataFrame(columns = ['date', 'summary'])
    for i in range(0, len(news_df), 1): 
            stories_df.loc[i,'date'] = news_df[i].created_at
            stories_df.loc[i,'summary'] = news_df[i].summary
            stories_df.loc[i, 'headline'] = news_df[i].headline
        
    stories_df
        

    # %%
    # Applying sentiment analysis to the news dataframe
    sentiment_df = get_sentiments(stories_df)
    sentiment_df.fillna(0)

    # %%
    # Normalizing sentiment df for easier concat with stock data
    sentiment_df.index = stories_df['date']
    sentiment_df.index = sentiment_df.index.normalize()
    sentiment_df.fillna(0, inplace=True)
    sentiment_df

    # %%
    # Resampling the sentiment df to aggregrate by day and take the mean of the sentiment scores
    sentiment_df_resampled = sentiment_df.resample('D', convention ='end').mean().fillna(0)
    sentiment_df_resampled.shape

    # %%
    sentiment_df_resampled.fillna(0, inplace=True)


    # %%
    # Normalizing stock dataframe and isolating the close values
    stock_df.index = stock_df.index.normalize()
    close_df = pd.DataFrame(stock_df["close"])

    # %%
    # Beginning our data processing for the stock close data with rolling averages, crossovers, etc.
    return_rolling_averages(close_df)

    # %%
    cross_df = return_crossovers(close_df)
    cross_df.shape

    # %%
    cross_signals=cross_df.sum(axis=1)

    # %%
    pct_change_df = close_df.pct_change()

    # %%
    cross_weighted_df = return_weighted_crossovers(close_df, pct_change_df)
    cross_weighted_df.shape

    # %%
    cross_signals_weighted = pd.DataFrame(cross_weighted_df.sum(axis=1))
    cross_signals_weighted.shape

    # %%
    # Retrieving volume data for concatenation with other features for signals dataframe
    volume_df = stock_df[['volume']]

    # %%
    # Concatenating dataframe with our cumulative signals
    signals_input_df = pd.DataFrame()
    signals_input_df = pd.concat([pct_change_df, cross_df, volume_df, pct_change_df, cross_signals, cross_signals_weighted, cross_weighted_df], axis=1).dropna()

    # %%
    signals_input_df.dropna(inplace=True)
    signals_input_df.shape

    # %%
    # Concatenating signals df with sentiment df
    signals_sentiment_df = pd.concat([signals_input_df, sentiment_df_resampled], axis=1).fillna(0)


    # %%
    # Using the finta library, we will implement more metrics for evaluation to be added to our signals dataframe
    from finta import TA
    finta_df = pd.DataFrame()

    finta_df['vama'] = TA.VAMA(stock_df)
    finta_df['rsi'] = TA.RSI(stock_df)
    finta_df['ao'] = TA.AO(stock_df)
    finta_df['ema'] = TA.EMA(stock_df)
    finta_df['evwma'] = TA.EVWMA(stock_df)
    finta_df['msd'] = TA.MSD(stock_df)
    finta_df['efi'] = TA.EFI(stock_df)
    finta_df['stochrsi'] = TA.STOCHRSI(stock_df)
    finta_df['tp'] = TA.TP(stock_df)
    finta_df['frama'] = TA.FRAMA(stock_df)
    finta_df['kama'] = TA.KAMA(stock_df)
    finta_df['hma'] = TA.HMA(stock_df)
    finta_df['obv'] = TA.OBV(stock_df)
    finta_df['cfi'] = TA.CFI(stock_df)
    finta_df['sma'] = TA.SMA(stock_df)
    finta_df['ssma'] = TA.SSMA(stock_df)
    finta_df['dema'] = TA.DEMA(stock_df)
    finta_df['tema'] = TA.TEMA(stock_df)
    finta_df['trima'] = TA.TRIMA(stock_df)
    finta_df['trix'] = TA.TRIX(stock_df)
    finta_df['smm'] = TA.SMM(stock_df)
    finta_df['zlema'] = TA.ZLEMA(stock_df)
    finta_df['vwap'] = TA.VWAP(stock_df)
    finta_df['smma'] = TA.SMMA(stock_df)
    finta_df['frama'] = TA.FRAMA(stock_df)
    finta_df['mom'] = TA.MOM(stock_df)
    finta_df['uo'] = TA.UO(stock_df)
    finta_df['vzo'] = TA.VZO(stock_df)
    finta_df['pzo'] = TA.PZO(stock_df)



    # %%


    # %%
    finta_df.fillna(0, inplace=True)
    finta_df.shape

    # %%
    finta_df.to_csv('finta.csv')

    # %%


    # %%
    signals_input_df = pd.concat([signals_input_df, finta_df], axis=1).dropna()

    # %%
    #Assigning our signals dataframe to X for train/test split
    X = signals_input_df
    X.shape

    # %%
    # Shifting our close df and comparing to original close df, we generate signals for whether or not the stock 
    # will go up for the following day.  We then convert the returned 0s to -1s for more robust predictions.
    y_signal = pd.DataFrame(((close_df["close"] > close_df["close"].shift()).shift(-1))*1, index = close_df.index)
    y_signal['close'] = np.where(y_signal['close'] == 0, -1, 1)

    # %%
    # Assigning the y_signals to our X index values
    y_signal = y_signal.loc[X.index]

    # %%
    # Displaying our y_signal data and stock close values to ensure the signals are correct.
    print(y_signal, close_df['close'])

    # %%
    # Assigning the y_signal['close'] data to y to create an array for train/test split
    y = np.array(y_signal['close'])

    # Checking shape of X and y objects to ensure they are aligned

    # Establishing train/test split
    train_num = int(X.shape[0] * 0.7)
    test_num = int(X.shape[0]-train_num)
        
    X_train = X[:train_num]
    X_test = X[-test_num:]

    y_train = y[:train_num]
    y_test = y[-test_num:]

    # %%
    # Scaling data for model processing
    scaler = StandardScaler()

    # %%
    scaler.fit(X_train)

    # %%
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Our first model is the 'svm' model from sklearn
    svm_model = svm.SVC()
    svm_model = svm_model.fit(X_train_scaled, y_train)
    svm_predictions = svm_model.predict(X_test_scaled)

    # Checking the accuracy of the 'svm' model
    
    print(classification_report(y_test, svm_predictions))
    

    # %%
    # Implementing logistical regression model from sklearn
    lr_model = LogisticRegression(max_iter=300)
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)
    

    # %%
    # Assessing accuracy of the lr model
    print(classification_report(y_test, lr_predictions))

    # %%
    # Gradient boosting classifier from sklearn
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model.fit(X_train_scaled, y_train) 
    gbc_predictions = model.predict(X_test_scaled)
    print(model.score(X_test_scaled, y_test))

    # %%
    # Implementing SCDClassifier, using 'log' loss metric for probabilistic prediction values
    sgdc = SGDClassifier(max_iter=1000, tol=0.01, loss='log')
    sgdc.fit(X_train_scaled, y_train)
    print(sgdc.score(X_train_scaled, y_train))
    sgdc_preds = sgdc.predict(X_test_scaled)
    print(classification_report(y_test, sgdc_preds))

    # %%
    shallow_preds = shallow_neural(X_train_scaled, y_train, X_test_scaled, y_test)

    deep_preds = deep_neural(X_train_scaled, y_train, X_test_scaled, y_test)

    # %%
    y_shallow = pd.DataFrame()
    y_shallow['signal'] = shallow_preds
    y_shallow['signal'][0] = 0.0


    # %%
    # Converting returned prediction values from the models in DataFrames, and then aligning indices 
    # and setting first and last signal to 0
    combined_preds = pd.DataFrame()
    svm_df = pd.DataFrame()
    lr_df = pd.DataFrame()
    gbc_df = pd.DataFrame()
    deep_df = pd.DataFrame()
    sgdc_df = pd.DataFrame()
    svm_df = pd.DataFrame(svm_predictions)
    lr_df = pd.DataFrame(lr_predictions)
    gbc_df = pd.DataFrame(gbc_predictions)
    deep_df = pd.DataFrame(deep_preds)
    sgdc_df = pd.DataFrame(sgdc_preds)
    svm_df.index = y_shallow.index
    lr_df.index = y_shallow.index
    gbc_df.index = y_shallow.index
    deep_df.index = y_shallow.index
    sgdc_df.index = y_shallow.index
    svm_df.iloc[0] = 0
    lr_df.iloc[0] = 0
    gbc_df.iloc[0] = 0
    deep_df.iloc[0] = 0
    sgdc_df.iloc[0] = 0
    svm_df.iloc[-1] = 0
    lr_df.iloc[-1] = 0
    gbc_df.iloc[-1] = 0
    deep_df.iloc[-1] = 0
    sgdc_df.iloc[-1] = 0

    # %%
    # Concat model dataframes
    combined_preds = pd.concat([y_shallow, svm_df, gbc_df, lr_df, deep_df, sgdc_df], axis=1)

    # %%
    combined_preds.columns = ['shallow_preds', 'svm_preds', 'gbc_preds', 'lr_preds', 'deep_preds', 'sgdc_preds']

    # %%
    # Calculating the mean prediction values across the models
    combined_preds['mean'] = (combined_preds['deep_preds'] + combined_preds['svm_preds'] + combined_preds['gbc_preds'] + combined_preds['lr_preds'] + combined_preds['shallow_preds'] + combined_preds['sgdc_preds'] ) / 6

    # %%
    combined_preds['signal'] = combined_preds['mean']
    combined_preds['signal'] = combined_preds['mean']
    combined_preds['signal'] = combined_preds['mean']

    # %%
    # In order to boost confidence levels and reduce deviation, we establish a buffer in setting our 
    #signals to -1 or 1 based on the mean of the models prediction values
    combined_preds['signal'] = combined_preds['signal'].apply(lambda x: -1 if x <= -0.3 else x)
    combined_preds['signal'] = combined_preds['signal'].apply(lambda x: 0 if x > -0.3 and x < 0.3 else x)
    combined_preds['signal'] = combined_preds['signal'].apply(lambda x: 1 if x >= 0.3 else x)

    # %%
    combined_preds.index = X_test.index
    combined_preds

    # %%
    # Establshing entry and exit points for our trades based on signals
    combined_preds['entry/exit'] = combined_preds['signal'].diff().fillna(0)
    combined_preds['close'] = close_df['close'].loc[combined_preds.index]
    combined_preds.dropna(inplace=True)

    # %%
    # Set initial capital
    initial_capital = initial_capital

    # Set the share size using initial capital and dividing by close price of the stock with a 10% buffer.
    share_size = (initial_capital / (combined_preds['close'].iloc[0] * 0.9)).round(2)

    combined_preds['position'] = (share_size * combined_preds['signal']).round(2)
    combined_preds



    # %%
    # Setting up our backtesting dataframe
    combined_preds['entry/exit Position'] = (combined_preds['position'].diff().fillna(0)).round(2)
    combined_preds['Portfolio Holdings'] = (combined_preds['close'] * combined_preds['position']).round(2)

    # %%
    combined_preds['Portfolio Cash'] = initial_capital - ((combined_preds['close'] * combined_preds['entry/exit Position']).cumsum()).round(2)
    combined_preds['Portfolio Total'] = (combined_preds['Portfolio Cash'] + combined_preds['Portfolio Holdings']).round(2)
    combined_preds['Portfolio Daily Returns'] = combined_preds['Portfolio Total'].pct_change().round(4)

    combined_preds['Portfolio Cumulative Returns'] = ((1 + combined_preds['Portfolio Daily Returns']).cumprod()).round(4)
    combined_preds.to_csv(Path(f'./reports/{ticker}_backtest.csv'))
    print(combined_preds)

    # %%
    # Establishing a pct_change df for comparison of model returns to actual returns
    pct_change = close_df[['close']].loc[combined_preds.index]
    pct_change = pct_change.pct_change().round(4)
    pct_change.dropna(inplace=True)
    pct_change

    cum_return = ((1+pct_change).cumprod() - 1) * 100
    cum_return_plot = cum_return.hvplot(title = f'{ticker} Cumulative Returns', label = 'Actual Returns', legend=True, yformatter='%.2f', ylabel = 'Cumulative Returns (%)')
    shallow_plot = (((combined_preds['Portfolio Cumulative Returns'] - 1))*100).hvplot(ylabel = 'Cumulative Returns (%)',label = 'Model Returns', legend = True, yformatter='%.2f')
    combined_plot = cum_return_plot * shallow_plot
    print(pct_change.std(), combined_preds['Portfolio Daily Returns'].std())
    portfolio_total = combined_preds['Portfolio Total'].hvplot(yformatter='%.2f', title=f'{ticker}')
    portfolio_cash = combined_preds['Portfolio Cash'].hvplot(yformatter='%.2f', title = f'{ticker}')
    #hvplot.save(portfolio_cash, filename=Path(f'./reports/{ticker}portfolio_cash.png'), fmt='png')
    hvplot.show(combined_plot + portfolio_cash + portfolio_total)
    #hvplot.save(combined_plot, filename = Path(f'./reports/{ticker}_combined_plot.png'), fmt='png')
    #hvplot.save(portfolio_total, filename= Path(f'./reports/{ticker}_portfolio_total.png'), fmt='png')


    return combined_preds

# %%



