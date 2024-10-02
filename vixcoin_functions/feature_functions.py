
from datetime import datetime, timedelta
import pandas as pd
from arch import arch_model
import yfinance as yf
import numpy as np


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


# Univariate retrievals (one ticker)
def retrieve_yahoo_close(ticker: str = 'spy', start_date: str = None, 
                         end_date: str = None) -> pd.Series:
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
        print(f"Processing Close {ticker}")
        price_df = yahoo_data.history(start=start_date, end=end_date).Close
        price_df.name = ticker

        if price_df.empty:
            raise Exception("No Prices.")
        return price_df
    except Exception as ex:
        print(f"Sorry, Data not available for '{ticker}': Exception is {ex}")


def retrieve_yahoo_volume(ticker: str = 'spy', start_date: str = '2007-07-02', 
                          end_date: str = '2021-10-01') -> pd.Series:
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
        print(f"Processing Volume {ticker}")
        price_df = yahoo_data.history(start=start_date, end=end_date).Volume
        price_df.name = ticker

        if price_df.empty:
            raise Exception("No Volume.")
        return price_df
    except Exception as ex:
        print(f"Sorry, Data not available for '{ticker}': Exception is {ex}")


def retrieve_yahoo_put_options_volume(ticker: str = 'spy', 
                                      date: str = '2007-07-02') -> pd.Series:
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
def retrieve_close_multiple_tickers(ticker_list: list, start_date: str, 
                                    end_date: str) -> pd.DataFrame:
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
        close_price = retrieve_yahoo_close(ticker, start_date=start_date, end_date=end_date)
        close_prices_dict[ticker] = close_price

    close_prices_df = pd.DataFrame(close_prices_dict)
    return close_prices_df