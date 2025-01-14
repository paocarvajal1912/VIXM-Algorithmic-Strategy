# Functions to load and clean data for the VIXM strategy
import pandas as pd
from typing import Dict, List
from feature_functions import (
    #garch_fit_and_predict, 
    correlation_filter,
    #retrieve_yahoo_close, retrieve_yahoo_volume, 
    #retrieve_close_multiple_tickers,
    retrieve_volume, 
    save_volume_to_csv, 
    load_demo_volume, 
    process_volume_data
)

# Calculation of security prices component X1
def cleanup_prices_and_get_vixm_price_and_return(
    close_prices_df_raw: pd.DataFrame,
    config: Dict,
    display_results: bool = False):
    """Get the first component X1 with clean prices.
    Missing prices are replace with the available
    previoous day price.

    Args:
        close_prices_df_raw (pd.DataFrame): A dataframe with raw security prices
        config (Dict): a dictionary containing the minimum correlation between prices
            of a ticker to be included
        display_results (bool, optional): Whether to display X1 last prices. 
            Defaults to False.

    Returns:
        Tuple: First, the first component X1
            Secondly, the clean vixm close price series
            Thirly, the clean vixm daily return series
    """
    # X1 (close prices) - Fill of missing values
    close_prices_df = close_prices_df_raw.ffill(axis='rows'
    )
    # Apply correlation filter to keep series with low correlation
    close_prices_component_df = correlation_filter(
        close_prices_df, min_corr=config['min_corr'], 
        key_column=config['key_column'], eliminate_first_column=False
    )
    # Filling price_t with price_t-1 if price_t not available
    close_prices_component_df = close_prices_component_df.ffill(axis='rows'
    )
    # First columns is not given back, so we take opportunity to rename them
    X1 = close_prices_component_df.add_suffix("_close").copy()

    # Additional time series for easy manipulation of VIXM close and returns
    vixm = X1['VIXM_close']
    vixm_ret = X1['VIXM_close'].pct_change()

    vixm = pd.DataFrame([vixm]).T
    vixm.columns = ['VIXM']  # 'vixm' will represent the close and 'vixm_ret' the return

    vixm_ret = pd.DataFrame([vixm_ret]).T
    vixm_ret.columns = ['VIXM_ret']

    if display_results:
        print("Last records of the first component X1:")
        print(X1.tail())

    print("Completed cleanup of close prices")
    print("Close prices included in First X component X1:")
    print(X1.columns)
    
    return X1, vixm, vixm_ret


# Calculation of security returns component X2
# Include returns that are (minimally) correlated with the VIXM return
def get_return_component(close_prices_df: pd.DataFrame,
    config: Dict,
    display_results: bool = False
    ):
    """Calculates the security returns and 

    Args:
        close_prices_df (pd.DataFrame): _description_
        config (Dict): _description_
        display_results (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    security_returns_df = close_prices_df.pct_change()
    security_returns_component_df = correlation_filter(                            
        security_returns_df, 
        min_corr=config["min_corr"], 
        key_column='VIXM', # VIXM col will have the return oof VIXM
        eliminate_first_column=False
    )

    # The generation of a suffix is necesary when evaluating feature importance for time series identification
    X2 = security_returns_component_df.add_suffix('_returns')

    if display_results:
        print("Last records of the second component X2:")
        print(X2.tail())
        
    print("Completed inclusion of returns")
    
    return X2

# Calculation of security volume component X3
def get_volume_component(
    ticker_list: List,
    config: Dict,
    vixm_ret: pd.Series,
    use_api: bool = True,
    display_results: bool = False
    ):
    """
    
    """
    volume_list = ticker_list[0:len(ticker_list)]

    if use_api == False:
        volume_df = load_demo_volume()
    else:
        volume_dict = retrieve_volume(volume_list, 
            config["start_date"], config["end_date"]
        )
        volume_df = process_volume_data(volume_dict)
    
    # Save and check if the save was successful
    if save_volume_to_csv(volume_df):
        print("Volume data saved successfully.")
    else:
        print("Failed to save volume data.")
        
    print("Completed retrieve of volume")

    volume_df_with_vixm = pd.concat([vixm_ret, volume_df], axis=1)

    volume_component_df = correlation_filter(
        volume_df_with_vixm, 
        min_corr=config["min_corr"], 
        key_column='VIXM_ret', 
        eliminate_first_column=True 
    )
    X3 = volume_component_df.add_suffix("_volume").copy()

    if display_results:
        print("Last columns of the component")
        print(X3.tail())
        print("Volume data included:")
        print(X3.columns)
        print("Volume data excluded due to low correlation:")
        tickers_w_suffix = [ticker + "_volume" for ticker in ticker_list]
        print(set(tickers_w_suffix) - set(X3.columns))
    print("Succesful calculation of security volume component X3" )
    return X3






