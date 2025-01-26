"""
Functions to generate visualizations for the
VIXM algorithmic strategy
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display

def plot_tickers_with_scaling(
    data, as_is_tickers, scaled_tickers=None, 
    scaled_factors=None, key: str = "prices",
    title=None,
    wide=10,
    height=5
    ):
    """
    Plot the relationship between multiple tickers' prices, with scaling for selected tickers.

    Args:
        data (dict): A dictionary containing price data as a DataFrame under the key 'prices'.
                     The DataFrame should have columns corresponding to the tickers.
        as_is_tickers (list of str): List of tickers to plot without scaling. Must be columns in the 'prices' DataFrame.
        scaled_tickers (list of str, optional): List of tickers to apply scaling to. Defaults to None.
        scaled_factors (list of float, optional): List of scaling factors corresponding to the `scaled_tickers`. Defaults to None.
        key: str = None

    Returns:
        None: Displays the graphs.
        
    # Example of calling the function
        plot_tickers_with_scaling(
            data,
            as_is_tickers=['SPY'],
            scaled_tickers=['VIXM', 'QQQ'],
            scaled_factors=[2, 1.5]
        )
    """
    if scaled_tickers is None:
        scaled_tickers = []
    if scaled_factors is None:
        scaled_factors = []

    # Ensure scaled_tickers and scaled_factors have the same length
    if len(scaled_tickers) != len(scaled_factors):
        raise ValueError("`scaled_tickers` and `scaled_factors` must have the same length.")

    # Combine as_is_tickers and scaled_tickers for plotting
    all_tickers = as_is_tickers + scaled_tickers

    # Creating subplots
    fig, ax = plt.subplots(1, 1, figsize=(wide, height))

    for i, ticker in enumerate(all_tickers):
        if key is not None:
            if ticker not in data[key].columns:
                raise ValueError(f"Ticker '{ticker}' not found in data['{key}'].")
        else:
            if ticker not in data.columns:
                raise ValueError(f"Ticker '{ticker}' not found in data.")


    # Get a color palette from matplotlib
    color_cycle = plt.cm.Set1.colors  # Tab10 colormap for distinct colors
    num_colors = len(color_cycle)

    # Check if the ticker needs scaling
    if len(scaled_tickers) > 0:
        for i, ticker in enumerate(scaled_tickers):
            factor_index = scaled_tickers.index(ticker)
            color = color_cycle[i % num_colors]
            factor = scaled_factors[factor_index]
            if key is not None:
                ax.plot(
                    data[key].index,
                    factor * data[key][ticker],
                    label=f"{factor} * {ticker} Close Price",
                    color=color
                )
            else:
                ax.plot(
                    data.index,
                    factor * data[ticker],
                    label=f"{factor} * {ticker} Close Price",
                    color=color
                )
    # Plot without scaling
    if len (as_is_tickers) > 0:
        for i, ticker in enumerate(as_is_tickers):
           color = color_cycle[(len(scaled_tickers) + i) % num_colors]
           if key is not None:
                ax.plot(
                    data[key].index,
                    data[key][ticker],
                    label=f"{ticker} Close Price",
                    color=color
                )
           else:
               ax.plot(
                    data.index,
                    data[ticker],
                    label=f"{ticker} Close Price",
                    color=color
                )
    if key is not None and title is None:
        ax.set_title(f"Plot of Ticker {key.capitalize()}")
    elif title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Plot of time series for {as_is_tickers} and {scaled_tickers}")
    ax.set_xlabel("Date")
    if key is not None:
        ax.set_ylabel(f"{key.capitalize()}")
    ax.legend()
    ax.grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    return True

def plot_profitability_ts(
    model, X_: pd.DataFrame, y_,
    vixm_ret: pd.DataFrame, data, config,
    is_train_window: bool = False
):
    """It plots curves to compare the profitability
    of VIXM Token in comparison with the VIXM ETN.
    """
    fit_ = model.predict(X_)
    fit_df = pd.DataFrame(fit_, index=X_.index)
    y_df = pd.DataFrame(y_, index=X_.index)
    vixm_returns_df = vixm_ret.loc[y_df.index.min() : y_df.index.max()]
    vixm_returns_df = vixm_returns_df.loc[y_df.index]

    # SPY as benchmark
    spy_in_period_returns = data['d_returns']['SPY'].loc[y_df.index.min() : y_df.index.max()]

    # To calculate returns, we should see if the signal predicts
    # correctvly the next day return. SO, to calculate the strategy returns
    # we shift again the prediction to the next day results
    y_df = y_df.shift(1)
    fit_df = fit_df.shift(1)
    
    ts_df = pd.concat(
        [vixm_returns_df, y_df, fit_df, spy_in_period_returns], 
        axis=1
    ).dropna()
    target_ticker = config["key_column"]
    ts_df.columns = [f'{target_ticker} Return', 'Correct Signal_shifted', 'Fit Signal_shifted', "SPY returns"]

    predicted_return = np.multiply(
        ts_df[f'{target_ticker} Return'], ts_df['Fit Signal_shifted']
    )
    max_return = np.multiply(
        ts_df[f'{target_ticker} Return'], ts_df['Correct Signal_shifted']
    )

    results_df = pd.concat([ts_df, predicted_return, max_return], axis=1)

    results_df.columns = [f'{target_ticker} Return', 'Correct Signal_shifted', 'Fit Signal_shifted', "SPY Return", 'VIXCOIN Token Return', "Max Return"]

    return_of_one_dollar_in_window_df = (1 + results_df[[f'{target_ticker} Return', "SPY Return", 'VIXCOIN Token Return', "Max Return"]]).cumprod()
    return_of_one_dollar_in_window_df.columns = [f'{target_ticker} ETN', "SPY Return", "VIXCOIN Token(in sample)", "Max Return"]

    if config['use_hvplot']:
        profitability_train_plot = return_of_one_dollar_in_window_df[[f'{target_ticker} ETN', "SPY Return", "VIXCOIN Token(in sample)"]].hvplot(
            title="Growth of $1 initial Investment in VIXCOIN Token",
            ylabel="Dollars $" ,
            width=1000,
            color=['green','red','blue']
        )
        display(profitability_train_plot)
    else:
        # Matplotlib visualization
        plt.figure(figsize=(10, 6))
        plt.plot(
            return_of_one_dollar_in_window_df.index,
            return_of_one_dollar_in_window_df["VIXCOIN Token(in sample)"],
            label="VIXCOIN Token (in sample)", color='blue'
        )
        plt.plot(
            return_of_one_dollar_in_window_df.index,
            return_of_one_dollar_in_window_df[f'{target_ticker} ETN'],
            label=f"{target_ticker} ETN", color='green'
        )
        plt.plot(
            return_of_one_dollar_in_window_df.index,
            return_of_one_dollar_in_window_df["SPY Return"],
            label=f"SPY return", color='red'
        )
        plt.xlabel("Date")
        plt.ylabel("Dollars ($)")
        if is_train_window:
            title = "In-Sample Growth of $1 Initial Investment in VIXCOIN Token"
        else:
            title = "Out-of-Sample Growth of $1 Initial Investment in VIXCOIN Token"
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
    cummulative_return_df = return_of_one_dollar_in_window_df.iloc[-1]
    return results_df, cummulative_return_df

def plot_cum_profit():
    print("Hello")


