# Vixm Feature Functions

import pandas as pd
from arch import arch_model
import yfinance as yf
from datetime import datetime
import numpy as np


def garch_fit_and_predict(series, ticker, horizon=1, p=1, q=1, o=1, print_series_name=False):
    #p=1,q=1, o=1 
    #series=returns_df['spy']
    #horizon=1
    """
    This function takes a series of returns, and get back the GJR-GARCH time series fit for the conditional volatility, 
    using one shock, and a t-student distribution of errors that accepts a skew.
    
    Args:
    series: a pandas Series containing the time series of returns for which to predict its unconditional volatility
    ticker: a string with the security name or the time series name for the output of the model
    horizon=1: integer. The number of future out of sample predictions of the model. It is set up to one per default.
    p,q,1: integer parameters of the GJR-GARCH model desired. All of them are defaulted to 1.
           For details on this parameters see:
           https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.base.ARCHModel.html
    print_series_name: indicator for output messages every time a series fit is completed. It is set to False per default.
                     When it's set to True will print the output, when set to False, it will not.
            For details in the model see:
            https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html
    
    
    Return:
    A pandas Series which contains the GJR-GARCH time series fit for the conditional volatility, where the predicted value for the next day is
    set to t, where t is the last date of the time series provided. The idea is to include the out-of-sample prediction (horizon) as a feature
    in time t to predict t+1, where t is the present moment. The returned series have already made the shift.
    """

    series=series.dropna()
    shock_skew_gm_model=arch_model(
                    series, 
                    p=p, q=q, o=o,
                    mean='constant',
                    vol='GARCH',
                    dist='skewt',
                    rescale=True
                    
    )
    if print_series_name==True:
        print(f"Processing series: {ticker}....." )
    
    
    #Fit GARCH model and predict

    results_shock_skew_gm=shock_skew_gm_model.fit(update_freq=0, disp="off")
    
    conditional_volatility=results_shock_skew_gm.conditional_volatility
    #summary               =results_shock_skew_gm.summary()
    forecast              =results_shock_skew_gm.forecast(horizon=1, reindex=False)

    # Prepare return of the series ready to include to X before shift
    serie_garch_before_shift=conditional_volatility.shift(-1)
    serie_garch_before_shift.iloc[-1,:]=forecast.variance.iloc[-1]

    return serie_garch_before_shift



def correlation_filter(series, min_corr=0.20, key_column='VIXM', eliminate_first_column=False):
    
    """
    
    
    """

    key_correlations=series.corr()[key_column]
    to_keep_columns=key_correlations[abs(key_correlations)>=min_corr].index
    filtered_series=series[to_keep_columns]
    
    if eliminate_first_column==True:
        filtered_series=filtered_series.iloc[:,1:]
    

    return filtered_series


def retrieve_yahoo_close(ticker = 'spy', start_date = '2011-02-01', end_date = '2021-11-29'):
    
    """
    This function retrieves from Yahoo Finance an individual time series of close prices from a given ticker
    If the close price for the ticker is not available, it provides an exception.
    
    Args:
    ticker: an string with the ticker to retrieve. Per default will retrieve the 'spy'
    start_date: the start date of the time series to retrieve in the format 'YYYY-MM-DD'. Per default will use '2007-07-02'
    end_date: the start date of the time series to retrieve in the format 'YYYY-MM-DD'. Per default will use '2021-10-01'
    
    Return:
    the time series of close price for the ticker as a Pandas Series with the Date and the close price time series
    """

    try:
        # get data based on ticker
        yahoo_data = yf.Ticker(ticker)
        print(f"Processing Close {ticker}")
        # select close price data using start date and end data
        price_df = yahoo_data.history(start=start_date, end=end_date).Close
        price_df.name = ticker
        # if no data retrieved raise exception
        if price_df.shape[0] == 0:
            raise Exception("No Prices.")
        return price_df
    # handle exception
    except Exception as ex:
        print(f"Sorry, Data not available for '{ticker}': Exception is {ex}")

        
# Define function to retrieve daily volume data from yahoo using ticker, start date and end date
def retrieve_yahoo_volume(ticker = 'spy', start_date = '2007-07-02', end_date = '2021-10-01'):

    """
    This function retrieves from Yahoo Finance a time series of traded volume from a given ticker
    If the volume for the ticker is not available, it provides an exception.
    
    Args:
    ticker: an string with the ticker to retrieve. Per default will retrieve the 'spy'
    start_date: the start date of the time series to retrieve in the format 'YYYY-MM-DD'. Per default will use '2007-07-02'
    end_date: the start date of the time series to retrieve in the format 'YYYY-MM-DD'. Per default will use '2021-10-01'
    
    Return:
    The time series of traded volume for the ticker as a Pandas Series including the Date
    """
    try:
        # get data based on ticker
        yahoo_data = yf.Ticker(ticker)
        print(f"Processing Volume {ticker}")
        # select data using start date and end data and calculate the daily return
        price_df = yahoo_data.history(start=start_date, end=end_date).Volume
        price_df.name = ticker
        # if no data retrieved raise exception
        if price_df.shape[0] == 0:
            raise Exception("No Prices.")
        return price_df
    # handle exception
    except Exception as ex:
        print(f"Sorry, Data not available for '{ticker}': Exception is {ex}")

# Define function to retrieve put daily volume data from yahoo using ticker, start date and end date
def retrieve_yahoo_put_options_volume(ticker = 'spy', date = '2007-07-02'):
    """
    This functions retrieves a time series of intraday volume for a given day.
    If the volume for the ticker is not available, it provides an exception.
    
    Args:
    ticker: an string with the ticker to retrieve. Per default will retrieve the options of the SPY with 'spy'
    date: the date for which the intraday series of options volume is retrieved, using the format 'YYYY-MM-DD'. Per default it will use '2007-07-02'
    
    Return:
    The intraday volume data as a Panda Series
    """

    try:
        # get data based on ticker
        yahoo_data = yf.Ticker(ticker)
        print(f"Processing put volume from {ticker}")
        # select data using start date and end data and calculate the daily return
        opts = yahoo_data.option_chain()
        price_df = opts.puts
        price_df.name = ticker
        price_df = price_df.volume
        # if no data retrieved raise exception
        if price_df.shape[0] == 0:
            raise Exception("No Prices.")
        return price_df
    # handle exception
    except Exception as ex:
        print(f"Sorry, Data not available for '{ticker}': Exception is {ex}")