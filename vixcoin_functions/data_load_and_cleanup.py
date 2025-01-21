# Functions to load and clean data for the VIXM strategy
import os
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import yaml

def upload_configuration(configpath: str = 'config.yml' , display_configuration: bool = False):
    """Upload and optionally display configuration"""
    
    # Load the YAML configuration file
    with open(configpath, 'r') as file:
        config = yaml.safe_load(file)

    # Display the configuration
    if display_configuration:
        print("Configuration loaded:")
        for key, value in config.items():
            print(f"{key}: {value}")
    print("Configuration uploaded succesfully")
    return config

def load_data(config: Dict, display_detailed_steps=False):
    """retrieve prices and volume from yahoo finance for ticker list
    The config file must have at least the following information: 
    ticker_list, start_date, and end_date.
    """
    prices_df = retrieve_close_multiple_tickers(
        config["ticker_list"],
        config["start_date"],
        config["end_date"],
        display_detailed_steps=display_detailed_steps
    )
    volume_df = retrieve_volume_multiple_tickers(
        config["ticker_list"],
        config["start_date"],
        config["end_date"],
        display_detailed_steps=display_detailed_steps
    )
    print("Raw data of prices and volume has been succesfully loaded")
    print("For the following tickers:")
    print("    Prices:", list(prices_df.columns))
    print("    Volume:", list(volume_df.columns))
    print()
    if display_detailed_steps:
        print("Prices Tail")
        print(prices_df)
        print(volume_df)
        print("Volume Tail")
        print(volume_df)
    
    return {
        'prices': prices_df,
        'volume': volume_df
    }
    
def clean_data(data):
    """Clean the data by filling missing values
    with the previous value in each column"""
    for key, df in data.items():
        data[key] = df.ffill(axis=0)
        print(f"{key} data succesfully cleaned")
    return data

def save_data(data, config):
    """Save the files to the directory in config file"""
    data_folder = config.get('data_folder', './data')
    os.makedirs(data_folder, exist_ok=True)
    for key, df in data.items():
        try:
            filename = f"clean_{key}.csv" 
            filepath = os.path.join(data_folder,filename)
            df.to_csv(filepath, index=True)
            print(f"Saved {filename} to {data_folder}")
        except Exception as e:
            print(f"Failed to save {key}: {e}")
            
def process_data_pipeline(config,display_detailed_steps=False):
    data = load_data(config,display_detailed_steps)
    data = clean_data(data)
    data['d_returns'] = data['prices'].pct_change()
    save_data(data, config)
    return data

# Univariate retrievals (one ticker)
def retrieve_yahoo_close(
    ticker: str = 'spy', start_date: str = None, 
    end_date: str = None, display_detailed_steps=True) -> pd.Series:
    """
    Retrieves the close price time series from Yahoo Finance.

    Args:
        ticker (str): Ticker to retrieve (default 'spy').
        start_date (str): Start date of the time series (format 'YYYY-MM-DD').
                          Defaults to 5 years before the end date if not provided.
        end_date (str): End date of the time series (format 'YYYY-MM-DD'). 
                        Defaults to yesterday's date if not provided.

    Returns:
        pd.Series: Time series of close prices.
    """
    # Set end_date to yesterday if not provided
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Set start_date to 5 years before the end_date if not provided
    if start_date is None:
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = (end_date_obj - timedelta(days=5*365)).strftime('%Y-%m-%d')

    try:
        yahoo_data = yf.Ticker(ticker)
        if display_detailed_steps:
            print(f"Processing Close {ticker}")
        price_df = yahoo_data.history(start=start_date, end=end_date).Close
        price_df.name = ticker

        if price_df.empty:
            raise Exception("No Prices.")
        return price_df
    except Exception as ex:
        print(f"Sorry, Data not available for '{ticker}': Exception is {ex}")


def retrieve_yahoo_volume(
    ticker: str = 'spy', start_date: str = '2007-07-02', 
    end_date: str = '2021-10-01', display_detailed_steps=True) -> pd.Series:
    """
    Retrieves the traded volume time series from Yahoo Finance.

    Args:
        ticker (str): Ticker to retrieve (default 'spy').
        start_date (str): Start date of the time series (format 'YYYY-MM-DD').
        end_date (str): End date of the time series (format 'YYYY-MM-DD').

    Returns:
        pd.Series: Time series of traded volume.
    """
    try:
        yahoo_data = yf.Ticker(ticker)
        if display_detailed_steps:
            print(f"Processing Volume {ticker}")
        price_df = yahoo_data.history(start=start_date, end=end_date).Volume
        price_df.name = ticker

        if price_df.empty:
            raise Exception("No Volume.")
        return price_df
    except Exception as ex:
        print(f"Sorry, Data not available for '{ticker}': Exception is {ex}")


def retrieve_yahoo_put_options_volume(
    ticker: str = 'spy', date: str = '2024-09-27') -> pd.Series:
    """
    Retrieves intraday put options volume for a given day.

    Args:
        ticker (str): Ticker to retrieve (default 'spy').
        date (str): Date for the intraday series (format 'YYYY-MM-DD').

    Returns:
        pd.Series: Intraday put options volume.
    """
    try:
        yahoo_data = yf.Ticker(ticker)
        print(f"Processing put volume from {ticker}")
        opts = yahoo_data.option_chain()
        price_df = opts.puts
        price_df.name = ticker
        price_df = price_df.volume

        if price_df.empty:
            raise Exception("No Volume.")
        return price_df
    except Exception as ex:
        print(f"Sorry, Data not available for '{ticker}': Exception is {ex}")


# Multi-tickers in a list retrieval
def retrieve_close_multiple_tickers(
    ticker_list: list, start_date: str, 
    end_date: str,
    display_detailed_steps=True
    ) -> pd.DataFrame:
    """
    Retrieves close prices from Yahoo Finance for a list of tickers.

    Args:
        ticker_list (list): List of tickers to retrieve.
        start_date (str): Start date of the time series (format 'YYYY-MM-DD').
        end_date (str): End date of the time series (format 'YYYY-MM-DD').

    Returns:
        pd.DataFrame: DataFrame with close prices for each ticker.
    """
    close_prices_dict = {}

    for ticker in ticker_list:
        close_price = retrieve_yahoo_close(
            ticker, start_date=start_date, end_date=end_date,
            display_detailed_steps=display_detailed_steps)
        close_prices_dict[ticker] = close_price

    close_prices_df = pd.DataFrame(close_prices_dict)
    return close_prices_df


def retrieve_volume_multiple_tickers(
    volume_list: List[str], start_date: str, end_date: str,
    display_detailed_steps=True) -> Dict:
    """Retrieves volume trades for a list of tickers."""
    volume_dict = {}
    for ticker in volume_list:
        volume = retrieve_yahoo_volume(
            ticker, start_date=start_date, end_date=end_date,
            display_detailed_steps=display_detailed_steps)
        volume_dict[ticker] = volume
    volume_df = pd.DataFrame(volume_dict)
    return volume_df


def load_demo_volume(filepath: str = "demo_data/adaboost_volume.csv") -> pd.DataFrame:
    """Loads volume data from a CSV file (demo mode)."""
    return pd.read_csv(filepath, index_col="Date", parse_dates=True, infer_datetime_format=True)


def process_volume_data(volume_dict: Dict) -> pd.DataFrame:
    """Processes raw volume data into a forward-filled DataFrame."""
    volume_df_raw = pd.DataFrame(volume_dict)
    return volume_df_raw.ffill(axis='rows')






