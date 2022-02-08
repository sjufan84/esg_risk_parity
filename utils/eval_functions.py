import pandas as pd
import numpy as np


# test to add up all exercises, no stops or changes to purchase/sell amount in place.
def results_trade_amount_nostop(start_money, close_df_test, trained_predictions, trade_amount):
    shares = 0
    money_on_hand = start_money
    shares_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    money_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    value_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
 
    
    for day in range(len(trained_predictions)):
        shares_df.iloc[day][0] = shares
        money_on_hand_df.iloc[day][0] = money_on_hand
        value_on_hand_df.iloc[day][0] = (shares * close_df_test.iloc[day]["close"]) + money_on_hand
        if trained_predictions.iloc[day][0] == 0:
            shares -= trade_amount / close_df_test.iloc[day]["close"]
            money_on_hand += trade_amount
        elif trained_predictions.iloc[day][0] == 1:
            shares += trade_amount / close_df_test.iloc[day]["close"]
            money_on_hand -= trade_amount
            
    return money_on_hand_df, shares_df, value_on_hand_df


# alternate test to add up all exercises, stops in place to prevent negative money or shares.
def results_trade_amount_stops(start_money, close_df_test, trained_predictions, trade_amount):
    shares = 0
    money_on_hand = start_money
    shares_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    money_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    value_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    
    for day in range(len(trained_predictions)):
        shares_df.iloc[day][0] = shares
        money_on_hand_df.iloc[day][0] = money_on_hand
        value_on_hand_df.iloc[day][0] = (shares * close_df_test.iloc[day]["close"]) + money_on_hand
        if (trained_predictions.iloc[day][0] == 0) & (shares - trade_amount / close_df_test.iloc[day]["close"] >= 0):
            shares -= trade_amount / close_df_test.iloc[day]["close"]
            money_on_hand += trade_amount
        elif (trained_predictions.iloc[day][0] == 1) & (money_on_hand - trade_amount >= 0):
            shares += trade_amount / close_df_test.iloc[day]["close"]
            money_on_hand -= trade_amount
            
    return money_on_hand_df, shares_df, value_on_hand_df



# alternate strategy. buy all on buy signal, sell all on sell signal, if available.
def buy_or_sell_all_if_available(start_money, close_df_test, trained_predictions):
    shares = 0
    money_on_hand = start_money
    shares_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    money_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    value_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    
    for day in range(len(trained_predictions)):
        shares_df.iloc[day][0] = shares
        money_on_hand_df.iloc[day][0] = money_on_hand
        value_on_hand_df.iloc[day][0] = (shares * close_df_test.iloc[day]["close"]) + money_on_hand
        if (trained_predictions.iloc[day][0] == 0) & (shares > 0):
            money_on_hand = shares * close_df_test.iloc[day]["close"]
            shares = 0
        elif (trained_predictions.iloc[day][0] == 1) & (money_on_hand > 0):
            shares = money_on_hand / close_df_test.iloc[day]["close"]
            money_on_hand = 0
    
    return money_on_hand_df, shares_df, value_on_hand_df



# alternate strategy. spend trade_percent of available money on buy signal, sell trade_percent of available shares on sell signal.

def buy_or_sell_trade_percent(trade_percent, start_money, close_df_test, trained_predictions):
    shares = 0
    money_on_hand = start_money
    shares_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    money_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    value_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    
    for day in range(len(trained_predictions)):
        shares_df.iloc[day][0] = shares
        money_on_hand_df.iloc[day][0] = money_on_hand
        value_on_hand_df.iloc[day][0] = (shares * close_df_test.iloc[day]["close"]) + money_on_hand
        if (trained_predictions.iloc[day][0] == 0) & (shares > 0):
            money_on_hand += (shares*trade_percent) * close_df_test.iloc[day]["close"]
            shares = shares*(1-trade_percent)
        elif (trained_predictions.iloc[day][0] == 1) & (money_on_hand > 0):
            shares += (money_on_hand*trade_percent) / close_df_test.iloc[day]["close"]
            money_on_hand = money_on_hand*(1-trade_percent)
            
    return money_on_hand_df, shares_df, value_on_hand_df





# test all SMA crossover signals with buy all/sell all as execution of signal
def sma_crossover_eval(start_money, cross_df, close_df):
    start_money_reset = start_money
    shares_reset = 0
    cross_cols = cross_df.columns
    
    shares_df = pd.DataFrame(np.zeros(shape=cross_df.shape), index=cross_df.index, columns = cross_cols)
    money_on_hand_df = pd.DataFrame(np.zeros(shape=cross_df.shape), index=cross_df.index, columns = cross_cols)
    value_on_hand_df = pd.DataFrame(np.zeros(shape=cross_df.shape), index=cross_df.index, columns = cross_cols)



    for col in cross_cols:
        money_on_hand = start_money_reset
        shares = shares_reset
        for day in range(len(cross_df)):
            shares_df[col].iloc[day] = shares
            money_on_hand_df[col].iloc[day] = money_on_hand
            value_on_hand_df[col].iloc[day] = (shares * close_df.iloc[day]["close"]) + money_on_hand
            if (cross_df[col].iloc[day] == -1) & (shares > 0):
                money_on_hand = shares * close_df.iloc[day]["close"]
                shares = 0
            elif (cross_df[col].iloc[day] == 1) & (money_on_hand > 0):
                shares = money_on_hand / close_df.iloc[day]["close"]
                money_on_hand = 0
    return value_on_hand_df