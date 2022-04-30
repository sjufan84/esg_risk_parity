# Machine Learning and Heirarchical Risk Parity ESG Portfolio Construction

Constructing a portfolio of crypto and stock assets utlizing ESG scores as well as machine learning models to predict buy / sell signals after establishing asset weights using hierarchical risk parity models.

## Technologies

In this project we are utilizing Python 3, Voila, Alpaca Trade API, Pandas, Holoviews, Numpy, Jupyter Lab and Plotly  

Voila -- allows you to convert a Jupyter Notebook into an interactive dashboard that allows you to share your work with others. It is secure and customizable, giving you control over what your readers experience.  
Alpaca Trade API -- allows for the simple access of historical stock and crypto data  
Holoviews -- a visualization library that includes hvplot and other tools to create simple and customizable charts  
Plotly -- another tool to generate plots

## Installation Guide

* Voila -- install via Pypi --  
 `pip install voila` in the command line  
 **For more information on Voila, visit their documentation site at https://voila.readthedocs.io/en/latest/index.html**

 * alpaca trade api -- install via Pypi --  
 `pip install alpaca-trade-api` in the command line  

 * holoviews -- install via Pypi --  
 `pip install holoviews` in the command line  

 * plotly --  
 `pip install holoviews` in the command line 

## Usage

  After ensuring all necessary libraries are installed, use Voila to visualize the underlying 'esg_parity_voila.ipynb'.  This can be done one of 2 ways:
  1) Directly from the command line -- `voila <path-to-notebook>`
  2) Using Jupyter Lab Extension, start jupyter lab server and then open a browser window with `<url-of-my-server>/voila`
  
  **Please note this will take some time depending on your system's specs and internet connection due to the API calls and the multiple ML models that are being run --  for digging in to the code behind the API calls and Machine learning models used, navigate to the utils folder containing the various scripts used**


## License

Licensed under the [MIT License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)  Copyright 2022 Dave Thomas.

#### For questions or more information please contact me at [sjufan84@gmail.com](mailto:sjufan84@gmail.com)
