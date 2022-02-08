# Initial imports
import datetime as dt
from dotenv import load_dotenv
import sys
import os
import pandas as pd
import requests
import json
from pprint import pprint


#import environment variables
load_dotenv()
ESG_api_key = os.getenv('ESG_API_KEY')
rapidapi_host = 'esg-environmental-social-governance-data.p.rapidapi.com'




# Constructing the API call body
url = "https://esg-environmental-social-governance-data.p.rapidapi.com/search"
headers = {
    'x-rapidapi-host': "esg-environmental-social-governance-data.p.rapidapi.com",
    'x-rapidapi-key': "ed5753e239msh35bbc80821a70dcp1cc1c1jsn993b485f9da5"
    }




# Function that takes in ticker and generates ESG score into a pandas dataframe
def get_esg_df(ticker):
    ESG_df = pd.DataFrame(index = [ticker], columns = ['Environment Grade', 'Governance Grade', 'Social Grade'])
    
    querystring = {'q' : ticker}
    ticker = ticker

    response = requests.request("GET", url, headers=headers, params=querystring).json()
    print(response[0])
    env_grade = response[0]['environment_grade']
    gov_grade = response[0]['governance_grade']
    social_grade = response[0]['social_grade']
    ESG_df.loc[ticker]['Environment Grade'] = env_grade
    ESG_df.loc[ticker]['Governance Grade'] = gov_grade
    ESG_df.loc[ticker]['Social Grade'] = social_grade  
    
    return ESG_df                


