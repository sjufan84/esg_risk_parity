import pandas as pd
import numpy as np
from pathlib import Path

# returns a dataframe with rolling average columns. moving average numbers were chosen from reported amounts of the most common ones used by other traders. 5, 10, 20, 50, 100, and 200, plus fibonacci numbers in the same range.
def return_rolling_averages(dataframe):
    windows = [2, 3, 5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200]
    for w in windows:
        dataframe[w]=dataframe["close"].rolling(w).mean()

# returns a dataframe with signals for each time a lower moving average crosses a higher length rolling average. the result is when a lower one passes a longer one, return signal plus one. when a lower length average dips below a longer one, return signal negative one.
def return_crossovers(dataframe):
    columns = dataframe.columns
    cross_df = pd.DataFrame(index=dataframe.index)
    for col in range(len(dataframe.columns)):
        for col2 in (range(col+1, len(dataframe.columns))):
            cross_df[str(columns[col]) + " to " + str(columns[col2])] = ((dataframe[columns[col2]] < dataframe[columns[col]]) & ((dataframe.shift()[columns[col2]] > dataframe.shift()[columns[col]]))) * 1 - ((dataframe[columns[col2]] > dataframe[columns[col]]) & ((dataframe.shift()[columns[col2]] < dataframe.shift()[columns[col]]))) * 1
    return cross_df

# returns a dataframe with signals for each time a lower moving average crosses a higher length rolling average. instead of returning a plus one or negative one signal, it returns the slope of the difference. input should be the pct_change_df
def return_weighted_crossovers(close_df, pct_change_df):
    columns = close_df.columns
    cross_weighted_df = pd.DataFrame(np.zeros(shape=close_df.shape[0]), index=close_df.index)
    for col in range(len(close_df.columns)):
        for col2 in (range(col+1, len(close_df.columns))):
            cross_weighted_df[str(columns[col]) + " to " + str(columns[col2]) + " weighted"] = (((close_df[columns[col2]] < close_df[columns[col]]) & ((close_df.shift()[columns[col2]] > close_df.shift()[columns[col]]))) * 1 - ((close_df[columns[col2]] > close_df[columns[col]]) & ((close_df.shift()[columns[col2]] < close_df.shift()[columns[col]]))) * 1) * (pct_change_df[columns[col]] - pct_change_df[columns[col2]]) * (((((close_df[columns[col2]] < close_df[columns[col]]) & ((close_df.shift()[columns[col2]] > close_df.shift()[columns[col]]))) * 1)-.5)*2)
    return cross_weighted_df


# returns a dataframe that converts all the close and moving averages to daily percent change values
def return_pct_change(dataframe):
    pct_change_df = dataframe.pct_change()
    return pct_change_df
    

def get_ticker_keywords(ticker):
    # should check if CSV exists, but for now counting on pre-processed tickers only

    df = pd.read_csv(Path('sentiment_analysis/Resources/news_data/' + ticker + '.csv'), usecols= ['Key_words'])
    keywords = df['Key_words'].unique()
    return keywords

