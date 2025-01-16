
from typing import List, Dict
import pandas as pd
from arch import arch_model
from data_load_and_cleanup import (
    garch_fit_and_predict, 
    correlation_filter,
    retrieve_yahoo_close, 
    retrieve_volume, 
    save_volume_to_csv, 
    load_demo_volume, 
    process_volume_data
)


def garch_fit_and_predict(series: pd.Series, ticker: str, horizon: int = 1, 
                          p: int = 1, q: int = 1, o: int = 1, 
                          print_series_name: bool = False) -> pd.Series:
    """
    Fits a GJR-GARCH model to a series of returns and predicts the conditional volatility.

    Args:
        series (pd.Series): Time series of returns for which to predict volatility.
        ticker (str): Ticker symbol or name for the series.
        horizon (int): Number of future predictions (default 1).
        p, q, o (int): Parameters of the GJR-GARCH model (default 1).
        print_series_name (bool): Prints series name if True (default False).

    Returns:
        pd.Series: GJR-GARCH time series fit for conditional volatility with future prediction.
    """
    series = series.dropna()
    shock_skew_gm_model = arch_model(
        series, p=p, q=q, o=o,
        mean='constant',
        vol='GARCH',
        dist='skewt',
        rescale=True
    )

    if print_series_name:
        print(f"Processing series: {ticker}...")

    # Fit GARCH model and predict
    results_shock_skew_gm = shock_skew_gm_model.fit(update_freq=0, disp="off")
    conditional_volatility = results_shock_skew_gm.conditional_volatility
    forecast = results_shock_skew_gm.forecast(horizon=horizon, reindex=False)

    # Prepare the series ready for inclusion in X before shifting
    serie_garch_before_shift = conditional_volatility.shift(-1)
    serie_garch_before_shift.iloc[-1] = forecast.variance.iloc[-1]

    return serie_garch_before_shift

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
    
    return X1, vixm, vixm_ret, close_prices_df

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
        security_returns_df.copy(), 
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
    
    return X2, security_returns_df

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

# Calculation of the GARCH component X4
def get_garch_component(
    ticker_list: List,
    config: Dict,
    security_returns_df: pd.DataFrame,
    display_results: bool = False
    ):
    """Calculates the GARCH conditionally volatility forecasting
    for each security.

    Args:
        use_api (bool, optional): Whether the Yahoo API should be use for the data.
            Defaults to True. If false will read from a csv file.
    """
    garch_series_df = pd.DataFrame()
    not_to_include = config['garch_not_to_include']

    for ticker in ticker_list:
        if ticker in not_to_include:
            continue

        garch_series_df[ticker] = garch_fit_and_predict(
            security_returns_df[ticker], ticker, 
            horizon=1, p=1, q=1, o=1, 
            print_series_name=display_results
        )

    X4 = garch_series_df.add_suffix("_garch")

    if display_results:
       print(X4.tail())

    print('GARCH Process fit and predictions completed for component X4')
    
    return X4


# Calculation of the return squares component X5
def get_return_squared(
    config: Dict,
    vixm_ret: pd.DataFrame,
    security_returns_df: pd.DataFrame,
    display_results: bool = False
    ):
    """Calculates the return squared of each ticker

    Args:
        config (Dict): configuration file containing the minimum 
            correlation of the squared returned with the vixm 
            squared return consider to keep the series. 
        vixm_ret (pd.Series): _description_
        security_returns_df (pd.DataFrame): _description_
        display_results (bool, optional): It will show the tail of the 
            resulting dataframe if True. Defaults to False.
    """
    returns_squared_df = security_returns_df ** 2
    vixm_ret2 = vixm_ret ** 2
    returns_squared_and_vixm_ret_df = pd.concat(
        [vixm_ret2,returns_squared_df], 
        axis=1
    )
    returns_squared_component_df = correlation_filter(
        returns_squared_and_vixm_ret_df, 
        min_corr=config["min_corr"], 
        key_column='VIXM_ret', # this is vixm ret squared
        eliminate_first_column=True
    )
    X5 = returns_squared_component_df.add_suffix(
        "_return_squared"
    )
    if display_results:
        print(X5.tail())
    print("Return squared calculation completed for componenent 5 (X5)")

    return X5

# Calculation of SPY volatility on varios windows
def get_ticker_volatilities(
    start_date_volatilities,
    vixm,
    ticker: str = 'spy',
    end_date = None,
    config = {},
    display_results: bool = False
):
    end_date = end_date if end_date is not None else config['end_date']
    close_price_ticker_df = retrieve_yahoo_close(
        ticker,
        start_date=start_date_volatilities,
        end_date=end_date
    )
    close_price_ticker_df.to_csv(f"demo_data/adaboost_{ticker}_data.csv", index=True)

    ticker_returns_df = close_price_ticker_df.pct_change()

    # Calculates calculation
    ticker_volatility = pd.DataFrame()
    windows_for_lag = config['windows_for_volatility_lags']

    for window_size in windows_for_lag:
        column_name = f"{window_size}_{ticker}_rolling_volatility"
        ticker_volatility[column_name] = ticker_returns_df.rolling(
            window=window_size).std()

    # Concatenate to vixm to uniform index
    X8 = pd.concat([vixm, ticker_volatility], axis=1)
    
    # Fill missing data, and delete vixm, that was  used to uniform the index
    X8 = X8.ffill()
    X8 = X8.iloc[:, 1:]

    # Setting for demo
    if display_results:
        print(X8.shape)
        print(X8.tail())
    print("Rolling volatilities component completed (X8)")
    
    return X8


# Calculation of day of the week effect
# 0: Monday, 6: Sunday
def get_day_of_week_component(
    close_prices_df: pd.DataFrame,
    display_results: bool = False
):
    """Add dummy features for each day of the week"""
    date_idx_name = close_prices_df.index.name
    # Calculate day of week for index dates (in "Date" column)
    day_of_week_df = pd.DataFrame(
        close_prices_df.index.dayofweek,
        close_prices_df.index
    )
    # Transform day numbers to dummies
    day_of_week_df = pd.concat(
        [day_of_week_df, pd.get_dummies(day_of_week_df[date_idx_name])],
        axis=1
    )
    # Drop "Date" col and rename dummy cols with day of the week
    day_of_week_df.drop(columns=[date_idx_name], inplace=True)
    day_of_week_df.columns = ["Mon", "Tue", "Wed","Thu","Fri"]

    if display_results:
        print("day_of_week_df.shape: ", day_of_week_df.shape)
        print("day_of_week_df.value_counts")
        print(day_of_week_df.value_counts())
        print("tail\n", day_of_week_df.tail())
    print("Calculation of day of the week effect component (X10) completed.")

    return day_of_week_df

# Calculation of the Month Effect Component
def get_month_component(
    close_prices_df: pd.DataFrame,
    display_results: bool = False
):
    """Add dummy features with the month of the index
    """
    date_idx_name = close_prices_df.index.name
    month_df = pd.DataFrame(
        close_prices_df.index.month,
        index=close_prices_df.index
    )
    # Transform columns to features columns signaling the month as boolean
    month_df = pd.concat(
        [month_df,pd.get_dummies(month_df[date_idx_name])],
        axis=1
    )
    # Drop "Date" col and rename dummy cols with the month
    month_df.drop(columns=[date_idx_name], inplace=True)
    month_df.columns = ["Jan", "Feb", "Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    if display_results:
        print("month_df.shape:", month_df.shape)
        print("month_df value_counts")
        print(month_df.value_counts())
        print("tail\n", month_df.tail())
    print("Calculation of day of the week effect component (X10) completed.")

    return month_df
