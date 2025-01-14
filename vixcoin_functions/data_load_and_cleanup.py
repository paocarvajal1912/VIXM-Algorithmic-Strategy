# Functions to load and clean data for the VIXM strategy
import pandas as pd
from feature_functions import (
    #garch_fit_and_predict, 
    correlation_filter,
    #retrieve_yahoo_close, retrieve_yahoo_volume, 
    #retrieve_close_multiple_tickers,
    #retrieve_volume, save_volume_to_csv, load_demo_volume, process_volume_data
)

def cleanup_prices_and_get_vixm_price_and_return(
    close_prices_df_raw,config,display_data=False):
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

    if display_data:
        print("Last record of the first component X1:")
        print(X1.tail())

    print("Completed cleanup of close prices")
    print("Close prices included in First X component X1:")
    print(X1.columns)
    
    return X1, vixm, vixm_ret





