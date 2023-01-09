# %%
#imports
import pandas as pd
import numpy as np


# Multiple types of models from sklearn
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC


import hvplot.pandas

from finta import TA

from pathlib import Path
import datetime as dt

# %%
# import functions
from alpaca.data.timeframe import TimeFrame
from utils.AlpacaFunctions import get_historical_dataframe, get_crypto_bars
from utils.data_process import return_rolling_averages
from utils.data_process import return_crossovers
from utils.data_process import return_weighted_crossovers
#from sentiment_functions import get_sentiments

def getStockModel(ticker, numDays, initial_capital):
    
    numDay = numDays
    timeframe = TimeFrame.Day
    today = dt.datetime.today()
    start = dt.datetime.today() - dt.timedelta(days=numDays)
    yesterday = dt.datetime.today() - dt.timedelta(days=1)
    end = yesterday
    limit = 5000
    symbol = ticker
    timeframe=TimeFrame.Day
    limit = 5000
    stock_df = pd.DataFrame()

    stock_df = get_historical_dataframe(symbol=symbol, start=start, end=end, timeframe=timeframe, limit=limit)
    stock_df.index = stock_df.index.droplevel(level=0)

        # Iterating through tickers to isolate and concat close data
        

# Normalizing stock dataframe and isolating the close values
    close_df = pd.DataFrame(stock_df["close"])

    # %%
    # Beginning our data processing for the stock close data with rolling averages, crossovers, etc.
    return_rolling_averages(close_df)

    # %%
    cross_df = return_crossovers(close_df)

    # %%
    cross_signals = cross_df.sum(axis=1)

    # %%
    pct_change_df = close_df.pct_change()

    # %%
    cross_weighted_df = return_weighted_crossovers(close_df, pct_change_df)

    # %%
    cross_signals_weighted = pd.DataFrame(cross_weighted_df.sum(axis=1))

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
    finta_df['smma'] = TA.SMMA(stock_df)
    finta_df['frama'] = TA.FRAMA(stock_df)
    finta_df['mom'] = TA.MOM(stock_df)
    finta_df['uo'] = TA.UO(stock_df)
    finta_df['vzo'] = TA.VZO(stock_df)
    finta_df['pzo'] = TA.PZO(stock_df)
    finta_df.fillna(0, inplace=True)

    # %%
    pct_change_df.rename(columns={'close': 'pct'}, inplace=True)
    #stock_df.drop(columns='symbol', inplace=True)

    # %%
    # Concatenating dataframe with our cumulative signals
    signals_input_df = pd.DataFrame()
    signals_input_df = pd.concat([stock_df, pct_change_df, cross_df, pct_change_df,
                                  cross_signals, cross_signals_weighted, cross_weighted_df, finta_df], axis=1).dropna()
    signals_input_df.columns = signals_input_df.columns.astype('str')

    # %%
    #Assigning our signals dataframe to X for train/test split
    X = signals_input_df.dropna()

    # %%
    # Shifting our close df and comparing to original close df, we generate signals for whether or not the stock
    # will go up for the following day.  We then convert the returned 0s to -1s for more robust predictions.
    y_signal = pd.DataFrame(
        ((close_df["close"] > close_df["close"].shift()).shift(-1))*1, index=close_df.index)
    y_signal['close'] = np.where(y_signal['close'] == 0, -1, 1)

    # %%
    # Assigning the y_signals to our X index values
    y_signal = y_signal.loc[X.index]

    # %%
    # Displaying our y_signal data and stock close values to ensure the signals are correct.
    #display(y_signal, close_df['close'])

    # %%
    # Assigning the y_signal['close'] data to y to create an array for train/test split
    y = np.array(y_signal['close'])

    # %%
    # Establishing train/test split
    train_num = int(X.shape[0] * 0.9)
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

    # %%
    # Our first model is the 'svm' model from sklearn
    svm_model = svm.SVC()
    svm_model = svm_model.fit(X_train_scaled, y_train)
    svm_predictions = svm_model.predict(X_test_scaled)

    # %%
    # Checking the accuracy of the 'svm' model
    svm_training_report = classification_report(y_test, svm_predictions, output_dict=True)
    svm_training_report = pd.DataFrame(svm_training_report)
    svm_training_report = pd.DataFrame(svm_training_report.iloc[2][0:2])
    svm_minus = svm_training_report.iloc[0, 0]
    svm_plus = svm_training_report.iloc[1, 0]
    svm_training_report.columns = ['svm']

    # %%
    knc = KNeighborsClassifier()
    knc_model = knc.fit(X_train_scaled, y_train)
    knc_predictions = knc_model.predict(X_test_scaled)
    knc_training_report = classification_report(y_test, knc_predictions, output_dict=True)
    knc_training_report = pd.DataFrame(knc_training_report)
    knc_training_report = pd.DataFrame(knc_training_report.iloc[2][0:2])
    knc_minus = knc_training_report.iloc[0, 0]
    knc_plus = knc_training_report.iloc[1, 0]
    knc_training_report.columns = ['knc']

    # %%
    nvc = NuSVC()
    nvc_model = nvc.fit(X_train_scaled, y_train)
    nvc_predictions = nvc_model.predict(X_test_scaled)
    nvc_training_report = classification_report(y_test, nvc_predictions, output_dict=True)
    nvc_training_report = pd.DataFrame(nvc_training_report)
    nvc_training_report = pd.DataFrame(nvc_training_report.iloc[2][0:2])
    nvc_minus = nvc_training_report.iloc[0, 0]
    nvc_plus = nvc_training_report.iloc[1, 0]
    nvc_training_report.columns = ['nvc']

    # %%
    # Implementing logistical regression model from sklearn
    lr_model = LogisticRegression(max_iter=300, verbose=True)
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_training_report = classification_report(y_test, lr_predictions, output_dict=True)
    lr_training_report = pd.DataFrame(lr_training_report)
    lr_training_report = pd.DataFrame(lr_training_report.iloc[2][0:2])
    lr_minus = lr_training_report.iloc[0, 0]
    lr_plus = lr_training_report.iloc[1, 0]
    lr_training_report.columns = ['lr']

    # %%
    rfc = RandomForestClassifier()
    rfc.fit(X_train_scaled, y_train)
    rfc_predictions = rfc.predict(X_test_scaled)
    rfc_training_report = classification_report(y_test, rfc_predictions, output_dict=True)
    rfc_training_report = pd.DataFrame(rfc_training_report)
    rfc_training_report = pd.DataFrame(rfc_training_report.iloc[2][0:2])
    rfc_minus = rfc_training_report.iloc[0, 0]
    rfc_plus = rfc_training_report.iloc[1, 0]
    rfc_training_report.columns = ['rfc']

    # %%
    # Gradient boosting classifier from sklearn
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train_scaled, y_train)
    gbc_predictions = model.predict(X_test_scaled)
    gbc_training_report = classification_report(y_test, gbc_predictions, output_dict=True)
    gbc_training_report = pd.DataFrame(gbc_training_report)
    gbc_training_report = pd.DataFrame(gbc_training_report.iloc[2][0:2])
    gbc_minus = gbc_training_report.iloc[0, 0]
    gbc_plus = gbc_training_report.iloc[1, 0]
    gbc_training_report.columns = ['gbc']

    # %%
    # Implementing SCDClassifier, using 'log' loss metric for probabilistic prediction values
    sgdc = SGDClassifier(max_iter=1000)
    sgdc.fit(X_train_scaled, y_train)
    sgdc_preds = sgdc.predict(X_test_scaled)
    sgdc_training_report = classification_report(y_test, sgdc_preds, output_dict=True)
    sgdc_training_report = pd.DataFrame(sgdc_training_report)
    sgdc_training_report = pd.DataFrame(sgdc_training_report.iloc[2][0:2])
    sgdc_minus = sgdc_training_report.iloc[0, 0]
    sgdc_plus = sgdc_training_report.iloc[1, 0]
    sgdc_training_report.columns = ['sgdc']

    # %%
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train_scaled, y_train)
    dtc_preds = dtc.predict(X_test_scaled)
    dtc_training_report = classification_report(y_test, dtc_preds, output_dict=True)
    dtc_training_report = pd.DataFrame(dtc_training_report)
    dtc_training_report = pd.DataFrame(dtc_training_report.iloc[2][0:2])
    dtc_minus = dtc_training_report.iloc[0, 0]
    dtc_plus = dtc_training_report.iloc[1, 0]
    dtc_training_report.columns = ['dtc']

    # %%
    lta = LinearDiscriminantAnalysis()
    lta.fit(X_train_scaled, y_train)
    lta_preds = lta.predict(X_test_scaled)
    lta_training_report = classification_report(y_test, lta_preds, output_dict=True)
    lta_training_report = pd.DataFrame(lta_training_report)
    lta_training_report = pd.DataFrame(lta_training_report.iloc[2][0:2])
    lta_minus = lta_training_report.iloc[0, 0]
    lta_plus = lta_training_report.iloc[1, 0]
    lta_training_report.columns = ['lta']

    # %%
    combined_accuracy = pd.concat([svm_training_report, gbc_training_report, sgdc_training_report, lta_training_report,
                                   nvc_training_report, lr_training_report, knc_training_report, rfc_training_report, dtc_training_report], axis=1)

    # %%
    combined_accuracy['mean'] = combined_accuracy.mean(axis=1)

    # %%
    svm_df = pd.DataFrame()
    svm_df = pd.DataFrame(svm_predictions, index=X_test.index)
    svm_df[0] = svm_df[0].apply(lambda x: x * svm_minus if x == -1 else x * svm_plus)
    svm_df

    # %%
    # Converting returned prediction values from the models in DataFrames, and then aligning indices
    # and setting first and last signal to 0
    combined_preds = pd.DataFrame()
    lr_df = pd.DataFrame(lr_predictions)
    lr_df[0] = lr_df[0].apply(lambda x: x * lr_minus if x == -1 else x * lr_plus)
    gbc_df = pd.DataFrame(gbc_predictions)
    gbc_df[0] = gbc_df[0].apply(lambda x: x * gbc_minus if x == -1 else x * gbc_plus)
    sgdc_df = pd.DataFrame(sgdc_preds)
    sgdc_df[0] = sgdc_df[0].apply(lambda x: x * sgdc_minus if x == -1 else x * sgdc_plus)
    dtc_df = pd.DataFrame(dtc_preds)
    dtc_df[0] = dtc_df[0].apply(lambda x: x * dtc_minus if x == -1 else x * dtc_plus)
    lta_df = pd.DataFrame(lta_preds)
    lta_df[0] = lta_df[0].apply(lambda x: x * lta_minus if x == -1 else x * lta_plus)
    rfc_df = pd.DataFrame(rfc_predictions)
    rfc_df[0] = rfc_df[0].apply(lambda x: x * rfc_minus if x == -1 else x * rfc_plus)
    knc_df = pd.DataFrame(knc_predictions)
    knc_df[0] = knc_df[0].apply(lambda x: x * knc_minus if x == -1 else x * knc_plus)
    nvc_df = pd.DataFrame(nvc_predictions)
    nvc_df[0] = nvc_df[0].apply(lambda x: x * nvc_minus if x == -1 else x * nvc_plus)
    lr_df.index = svm_df.index
    gbc_df.index = svm_df.index
    sgdc_df.index = svm_df.index
    dtc_df.index = svm_df.index
    lta_df.index = svm_df.index
    rfc_df.index = svm_df.index
    knc_df.index = svm_df.index
    nvc_df.index = svm_df.index

    # %%
    # Concat model dataframes
    combined_preds = pd.concat([svm_df, gbc_df, lr_df, sgdc_df, dtc_df, lta_df, rfc_df, knc_df, nvc_df], axis=1)

    # %%
    combined_preds.columns = ['svm_preds', 'gbc_preds', 'lr_preds', 'sgdc_preds',
                              'dtc_preds', 'lta_preds', 'rfc_preds', 'knc_preds', 'nvc_preds']

    # %%
    # Calculating the mean prediction values across the models
    combined_preds['mean'] = (combined_preds['svm_preds'] + combined_preds['gbc_preds'] + combined_preds['lr_preds'] + combined_preds['sgdc_preds'] +
                              combined_preds['dtc_preds'] + combined_preds['lta_preds'] + combined_preds['rfc_preds'] + combined_preds['knc_preds'] + combined_preds['nvc_preds']) / 9

    # %%
    combined_preds['signals'] = combined_preds['mean']

    # %%
    # In order to boost confidence levels and reduce deviation, we establish a buffer in setting our
    #signals to -1 or 1 based on the mean of the models prediction values
    combined_preds['signals'] = combined_preds['signals'].apply(lambda x: -1 if x <= -0.2 else x)
    combined_preds['signals'] = combined_preds['signals'].apply(lambda x: 0 if x > -0.2 and x < .2 else x)
    combined_preds['signals'] = combined_preds['signals'].apply(lambda x: 1 if x >= 0.2 else x)

    # %%
    combined_preds.index = X_test.index

    # %%
    combined_preds.iloc[0, 10] = 0
    combined_preds.iloc[-1, 10] = 0

    # %%
    # Establshing entry and exit points for our trades based on signals
    combined_preds['entry/exit'] = combined_preds['signals'].diff().fillna(0)
    combined_preds['close'] = close_df['close'].loc[combined_preds.index]
    combined_preds.dropna(inplace=True)

    # %%
    # Set initial capital
    initial_capital = initial_capital

    # Set the share size using initial capital and dividing by close price of the stock with a 10% buffer.
    share_size = (initial_capital /
                  (combined_preds['close'] * 1.1)).round(2)

    combined_preds['position'] = (
        share_size * (combined_preds['signals'])).round(2)
    combined_preds.head(50)

    # %%
    # Setting up our backtesting dataframe
    combined_preds['entry/exit Position'] = (
        combined_preds['position'].diff().fillna(0)).round(2)
    combined_preds['Portfolio Holdings'] = (
        combined_preds['close'] * combined_preds['position']).round(2)

    # %%
    combined_preds['Portfolio Cash'] = initial_capital - \
        ((combined_preds['close'] *
          combined_preds['entry/exit Position']).cumsum()).round(2)
    combined_preds['Portfolio Total'] = (
        combined_preds['Portfolio Cash'] + combined_preds['Portfolio Holdings']).round(2)
    combined_preds['Portfolio Daily Returns'] = combined_preds['Portfolio Total'].pct_change(
    ).round(4)

    combined_preds['Portfolio Cumulative Returns'] = (
        (1 + combined_preds['Portfolio Daily Returns']).cumprod()).round(4)

    # %%
    # Establishing a pct_change df for comparison of model returns to actual returns
    pct_change = close_df[['close']].loc[combined_preds.index]
    pct_change = pct_change.pct_change().round(4)
    pct_change.dropna(inplace=True)

    # %%
    cum_return = ((1+pct_change).cumprod() - 1) * 100
    cum_return_plot = cum_return.hvplot(
        title=f'{ticker} Cumulative Returns', label='Actual Returns', legend=True, yformatter='%.2f', ylabel='Cumulative Returns (%)')
    shallow_plot = (((combined_preds['Portfolio Cumulative Returns'] - 1))*100).hvplot(
        ylabel='Cumulative Returns (%)', label='Model Returns', legend=True, yformatter='%.2f')
    combined_plot = cum_return_plot * shallow_plot
    portfolio_total = combined_preds['Portfolio Total'].hvplot(
        yformatter='%.2f', title=f'{ticker}')
    portfolio_cash = combined_preds['Portfolio Cash'].hvplot(
        yformatter='%.2f', title=f'{ticker}')
    #hvplot.save(portfolio_cash, filename=Path(f'./reports/{ticker}portfolio_cash.png'), fmt='png')
    hvplot.show(combined_plot + portfolio_cash + portfolio_total)
    #hvplot.save(combined_plot, filename = Path(f'./reports/{ticker}_combined_plot.png'), fmt='png')
    #hvplot.save(portfolio_total, filename= Path(f'./reports/{ticker}_portfolio_total.png'), fmt='png')

    return combined_preds[['mean', 'signals', 'entry/exit', 'close', 'position', 'entry/exit Position', 'Portfolio Holdings', 'Portfolio Cash', 'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns']]




