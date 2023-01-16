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



