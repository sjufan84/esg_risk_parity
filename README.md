# Machine Learning and Heirarchical Risk Parity ESG Portfolio Construction

Constructing a portfolio of crypto and stock assets utlizing ESG scores as well as machine learning models to predict buy / sell signals after establishing asset weights using hierarchical risk parity models.

## Technologies

In this project we are utilizing Python 3, Voila, Alpaca Trade API, Pandas, Holoviews, Numpy, Jupyter Lab and Plotly  

Voila -- allows you to convert a Jupyter Notebook into an interactive dashboard that allows you to share your work with others. It is secure and customizable, giving you control over what your readers experience.  
Alpaca Trade API -- allows for the simple access of historical stock and crypto data  
Holoviews -- a visualization library that includes hvplot and other tools to create simple and customizable charts  
Plotly -- another tool to generate plots
scikit-learn -- Machine Learning in Python.  Please visit their [Website](https://scikit-learn.org/stable/) for more information.
Finta -- Common financial technical indicators implemented in Pandas
Riskfolio -- Quantitative Strategic Asset Allocation, Easy for Everyone.  Visit their [website](https://riskfolio-lib.readthedocs.io/en/latest/index.html) for more information.

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
 
 * Finta -- 
 `pip install finta` in the command line  
 
 * Riskfolio --  
 `pip install riskfolio-lib` in the command line

 
 ## Introduction  
 
  ESG investing has become a central focus for many investors as they become more cognizant of the social impact of companies as well as there risk/return profiles.  Crytpocurrency investing has also exploded but the environmental impact of particular blockchains is being questioned as the amount of energy to mine these chains is extremely intense.  By creating a way to incorporate cryptocurrency investments with highly rated ESG stocks, one can theoretically try to offset some of the environmental concerns associated with crypto mining.  
  
  Machine learning is also a popular way to try to predict buy and sell signals in the market, and their are myriad models that can be used to that end.  In this project we have implemented several different models and weighted them according to their accuracy metrics.  These models include from sklearn import linear regression, gradient boosting, decision trees, and random forest among others. By incorporating standard market technical indicators with these models we are able to present an alternate methodology for portfolio management aside from typical buy and hold strategies.  
  
  Lastly, risk parity is one popular way to construct portfolios with a balanced risk profile to mitigate risk.  In this project we utilize the Riskfolio python library to implement a straightforward model to balance our capital allocation using Hierarchical Clustering Portfolio Optimization.  After allocating our initial capital with the calculated portfolio weights, we then run our ML models and compare the results to the actual returns and present via plotly and hvplot. We also run Monte Carlo Simulations to project out potential returns and the resulting probability distribution. 

## Usage

After ensuring all necessary libraries are installed, use Voila to visualize the underlying 'esg_parity_voila.ipynb'.  This can be done one of 2 ways:
1) Directly from the command line -- `voila <path-to-notebook>`
2) Using Jupyter Lab Extension, start jupyter lab server and then open a browser window with `<url-of-my-server>/voila`
  
  **Please note this will take some time depending on your system's specs and internet connection due to the API calls and the multiple ML models that are being run --  for digging in to the code behind the API calls and Machine learning models used, navigate to the utils folder containing the various scripts used**


## Licenses

Riskfolio -- Copyright (c) 2020-2022, Dany Cajas All rights reserved.

**Riskfolio Disclaimer --**  

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Licensed under the [MIT License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)  Copyright 2022 Dave Thomas.

#### For questions or more information please contact me at [sjufan84@gmail.com](mailto:sjufan84@gmail.com)
