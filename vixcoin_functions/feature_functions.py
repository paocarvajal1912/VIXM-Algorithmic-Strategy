
"""
Functions to generate all feature components of the
VIXM algoritmic strategy
"""
from typing import List, Dict
import pandas as pd
from arch import arch_model

def correlation_filter(
    series: pd.DataFrame, 
    min_corr: float = 0.20, 
    key_column: str = 'VIXM', 
    eliminate_first_column: bool = False) -> pd.DataFrame:
    """
    Filters series that do not meet the minimum correlation with the key column.

    Args:
        series (pd.DataFrame): DataFrame with time series to be filtered.
        min_corr (float): Minimum correlation threshold (default 0.20).
        key_column (str): Column name to measure correlation against.
        eliminate_first_column (bool): Whether to exclude the first column (default False).

    Returns:
        pd.DataFrame: Filtered DataFrame with columns meeting the correlation threshold.
    """
    key_correlations = series.corr()[key_column]
    to_keep_columns = key_correlations[abs(key_correlations) >= min_corr].index
    filtered_series = series[to_keep_columns]

    if eliminate_first_column:
        filtered_series = filtered_series.iloc[:, 1:]

    return filtered_series


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

def get_min_valid_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    Identify the minimum date where all time series in a DataFrame have valid data.

    Args:
        df (pd.DataFrame): A DataFrame with time series data. The index should be of datetime type.

    Returns:
        pd.Timestamp: The earliest date where all columns have valid data (no NaN values).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    valid_rows = df.dropna()
    min_date = valid_rows.index.min()

    return min_date


# Calculation of security prices component X1
def get_prices_component(
    close_prices_df: pd.DataFrame,
    config: Dict,
    display_results: bool = False):
    """Get the first component X1 with clean prices and 
    correlation clean up.

    Args:
        close_prices_df (pd.DataFrame): A dataframe with raw security prices
        config (Dict): a dictionary containing the minimum correlation between prices
            of a ticker to be included
        display_results (bool, optional): Whether to display X1 last prices. 
            Defaults to False.

    Returns:
        X1 (pandas dataframe): The first component X1 
    """

    # Apply correlation filter to keep series with low correlation
    prices_component_df = correlation_filter(
        close_prices_df, min_corr=config['min_corr'], 
        key_column=config['key_column'], 
        eliminate_first_column=False
    )
    # First columns is not given back, so we take opportunity 
    # to rename them
    X1 = prices_component_df.add_suffix("_close").copy()
    key_ticker_price_col_name = f"{config['key_column']}_close"
    
    # Additional time series for easy manipulation of VIXM close and returns
    vixm = X1[key_ticker_price_col_name]
    vixm_ret = X1[key_ticker_price_col_name].pct_change()

    vixm = pd.DataFrame([vixm]).T
    vixm.columns = [config['key_column']]  # VIXM will be the name, 'vixm' will represent the close and 'vixm_ret' the return
    vixm_ret = pd.DataFrame([vixm_ret]).T
    
    vixm_ret.columns = [f"{config['key_column']}_ret"] #VIXM_ret

    if display_results:
        print("Last records of the first component X1:")
        print(X1.tail())

        print("Close prices included in First X component X1:")
        print(X1.columns)
        print("Close prices excluded due to correlation filter:")
        print( set(close_prices_df.columns) - set(X1.columns))

    print("Calculation of price component completed")
    return X1, vixm, vixm_ret

# Calculation of security returns component X2
# Include returns that are (minimally) correlated with the VIXM return
def get_return_component(
    daily_returns_df: pd.DataFrame,
    config: Dict,
    display_results: bool = False
    ):
    """Filter returns that are relevant

    Args:
        dily_returns_df (pd.DataFrame): a dataframe containing daily returns
            in its columns, and the ticker as column names
        config (Dict): a dictionary containing the value of the minimum correlation
            needed between a security returns and the VIXM return
            in order to include in the output.
        display_results (bool, optional): wheter to show outputs
            of the last records, the column names, and the excluded
            securities due too correlation.

    Returns:
        pd.DataFrame: a dataframe with the returns of securities
            that have some minimal correlation with VIXM returns.
    """
    security_returns_component_df = correlation_filter(                            
        daily_returns_df.copy(), 
        min_corr=config["min_corr"], 
        key_column='VIXM', # VIXM col will have the return oof VIXM
        eliminate_first_column=False
    )

    # The generation of a suffix is necesary when evaluating feature importance for time series identification
    X2 = security_returns_component_df.add_suffix('_ret')

    if display_results:
        print("Last records of the second component X2:")
        print(X2.tail())
        print("Returns included in second component X2:")
        print(X2.columns)
        print("Returns excluded due to correlation filter:")
        print( set(daily_returns_df.columns) - set(X2.columns))
        
    print("Calculation of returns component completed")
    
    return X2

# Calculation of security volume component X3
def get_volume_component(
    volume_df: pd.DataFrame,
    config: Dict,
    vixm_ret: pd.Series,
    display_results: bool = False
    ):
    """Filter volumes of securities having a minimal correlation
    over returns with VIXM.
    Returns a dataframe with volumne, and column names suffixed
    with the word '_volume'"""
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
        print("Volume data excluded due to low correlation with vixm_ret:")
        tickers_w_suffix = [ticker + "_volume" for ticker in volume_df.columns]
        print(set(tickers_w_suffix) - set(X3.columns))
    print("Succesful calculation of security volume component X3" )
    return X3

# Calculation of the GARCH component X4
def get_garch_component(
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
    
    if display_results:
        print("Tickers not to be included per config: ", not_to_include)

    for ticker in security_returns_df.columns:
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
        vixm_ret (pd.Series): series with daily returns of the VIXM ETF
        security_returns_df (pd.DataFrame): dataframe with the returns of tickers
            on config
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
    returns: pd.DataFrame,
    ticker: str = 'SPY',
    config: Dict = {},
    display_results: bool = False
):
    """calculates volatilities for several different lags
    for the SPY or desired ticker."""
    
    ticker_returns_df = returns[ticker]
    
    # Calculates calculation
    ticker_volatility = pd.DataFrame()
    windows_for_lag = config['windows_for_volatility_lags']

    for window_size in windows_for_lag:
        column_name = f"{window_size}_{ticker}_rolling_volatility"
        ticker_volatility[column_name] = ticker_returns_df.rolling(
            window=window_size).std()

    # Concatenate to vixm to uniform index
    X6 = pd.concat([vixm, ticker_volatility], axis=1)
    
    # Fill missing data, and delete vixm, that was  used to uniform the index
    X6 = X6.ffill()
    X6 = X6.iloc[:, 1:]

    # Setting for demo
    if display_results:
        print(X6.shape)
        print(X6.tail())
    print("Rolling volatilities component completed (X6)")
    
    return X6


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
        print("day_of_week_df value_counts:")
        print(day_of_week_df.value_counts())
        print("\nDataframe Tail\n", day_of_week_df.tail())
    print("Calculation of day of the week effect component (X10) completed.")

    return day_of_week_df

# Calculation of the Month Effect Component
def get_month_component(
    close_prices_df: pd.DataFrame,
    display_results: bool = False
):
    """Add dummy features with the month of the index"""
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

# Concatenating all components to get X
def get_X(
    X_components: Dict, 
    config: Dict,
    inclusion_list: List = ['all']):
    """Concatenate the components to get the fetures matrix X"""
    X_old = pd.DataFrame()
    if inclusion_list == ['all']:
        inclusion_list = list(config['component_names'].keys())
        
    for key, df in X_components.items():
        if key in inclusion_list:
            X = pd.concat([X_old, df], axis=1)
            print(f"Component {key}: {config['component_names'][key]} included")
            X_old = X
    # Eliminating nulls for return calculatinos or lag windows
    min_valid_date = get_min_valid_date(X)
    X = X.loc[min_valid_date:]
    
    return X

def setup_signal(X, config, display_results=False):
    """Generate the trading signals 1 (entry for one day) 
    or 0 (do not enter)"""
    XY = X
    XY["Signal"] = 0.0
    XY.loc[(XY['VIXM_ret'] >= config['threshold']), 'Signal'] = 1

    # Define the target set y using the Signal column
    y = XY[["Signal"]]

    if display_results:
        print(XY[["Signal"]])
        print("XY[['Signal']] value_counts")
        print(XY[["Signal"]].value_counts())
        print("XY.shape: ", XY.shape)
        print(pd.concat([XY['VIXM_ret'],y], axis=1))
    
    return y, XY

 

