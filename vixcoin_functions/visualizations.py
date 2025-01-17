import matplotlib.pyplot as plt

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
                    color='orange'
                )
            else:
                ax.plot(
                    data.index,
                    factor * data[ticker],
                    label=f"{factor} * {ticker} Close Price",
                    color='orange'
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


