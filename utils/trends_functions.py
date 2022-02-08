import pandas as pd
from pathlib import Path

def concat_trends(ticker, df):
    Path = './sentiment_analysis/Resources/google_trends/{}.csv'.format(ticker)
    trends_data_df = pd.read_csv(Path, index_col = 'Date', parse_dates = True, infer_datetime_format = True)
    trends_data_df.index = trends_data_df.index.tz_localize(tz="America/New_York")
    

    trends_data_df = df.assign(key=df.index.normalize()).merge(trends_data_df, left_on='key', right_index=True, how='left').drop('key', 1)
    
    trends_data_df.dropna()
    
    return trends_data_df

