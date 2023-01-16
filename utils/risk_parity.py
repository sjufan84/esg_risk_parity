import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import riskfolio as rp

def get_parity_model(Y, rm):
    

    # Building the portfolio object
   

    port = rp.HCPortfolio(returns=Y)
    st.write(Y)

    # Create a dataframe with the below risk measure values:
    
    obj = 'Sharpe'
    

    model='HRP' # Could be HRP or HERC
    codependence = 'pearson' # Correlation matrix used to group assets in clusters
    rm = rm # Risk measure used, this time will be variance
    rf = 3 # Risk free rate
    linkage = 'single' # Linkage method used to build clusters
    max_k = 10 # Max number of clusters used in two difference gap statistic, only for HERC model
    leaf_order = True # Consider optimal order of leafs in dendrogram
    obj = obj
    l=2

    
    w = port.optimization(model=model,
        codependence=codependence,
        rm=rm,
        rf=rf,
        linkage=linkage,
        max_k=max_k,
        leaf_order=leaf_order,
        obj = obj,
        l=l
    )

    

    weights_df = pd.DataFrame(w.round(4))

    # %% [markdown]
    # #### After generating our portfolio weights we plot them using a pie chart as well as bar for visualization purposes

    # %%
    # Plotting the composition of the portfolio

    ax = px.pie(weights_df, title='Portfolio composition', names = list(weights_df.index), values = 'weights')
    st.plotly_chart(ax)

    # %%
    ax = px.bar(weights_df.sort_values(by='weights'), title='Portfolio composition', y='weights')
    st.plotly_chart(ax)

    
    return w
