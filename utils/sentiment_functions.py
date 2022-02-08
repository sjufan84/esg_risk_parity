import pandas as pd
from pathlib import Path

def concat_sentiment(ticker, df, twitter = True, gnews = True):
    Path = 'sentiment_analysis/Resources/Combined Sentiment signals/{}.csv'.format(ticker)
    sentiment_df = pd.read_csv(Path, index_col = 'Date', parse_dates = True, infer_datetime_format = True)
    sentiment_df.index = sentiment_df.index.tz_localize(tz="America/New_York")
    

    signals_with_sentiment_df = df.assign(key=df.index.normalize()).merge(sentiment_df, left_on='key', right_index=True, how='left').drop('key', 1)
    
    signals_with_sentiment_df.dropna()
    if(twitter == False):
        signals_with_sentiment_df = signals_with_sentiment_df.drop(columns = ['tw_subj_score', 'tw_simi_score', 'tw_vader_score','tw_scores_flair'])
        
    if(gnews == False):
        signals_with_sentiment_df = signals_with_sentiment_df.drop(columns = ['subj_score', 'simi_score', 'vader_score','scores_flair'])
    
    return signals_with_sentiment_df

