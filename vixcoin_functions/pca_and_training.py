
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def split_training_test_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    training_period_months: int = 12,
    display_results: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits a time-series dataset into training and testing windows.

    Args:
        X (pd.DataFrame): DataFrame containing features with a datetime index.
        y (pd.DataFrame): DataFrame or Series containing target signals.
        training_period_months (int): Number of months for the training period.
        display_results (bool): Print additional information if True (default True).

    Returns:
        Tuple: X_train, y_train, X_test, y_test.
    """
    training_begin = pd.to_datetime(X.index.min())
    training_end = training_begin + DateOffset(months=training_period_months)

    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]
    X_test = X.loc[training_end + DateOffset(days=1):]
    y_test = y.loc[training_end + DateOffset(days=1):]

    if display_results:
        print(f"Training period: {training_begin} to {training_end}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print("y_train value counts:")
        print(y_train.value_counts())
    print("Time series data split completed \n")

    return X_train, y_train, X_test, y_test


def scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = 'MinMaxScaler',
    display_results: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales the training and testing datasets.

    Args:
        X_train (pd.DataFrame): Training set features.
        X_test (pd.DataFrame): Testing set features.
        scaler_type (str): Type of scaler ('MinMaxScaler' or 'StandardScaler').
        tuning_mode (bool): Print additional information if True (default True).

    Returns:
        Tuple: Scaled X_train and X_test as numpy arrays.
    """
    scaler = MinMaxScaler() if scaler_type == 'MinMaxScaler' else StandardScaler()

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if display_results:
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
    
    print("Data scaled succesfully, based on trained data.")

    return X_train_scaled, X_test_scaled


def adaboost_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    idx: pd.Index,
    num_pca_components: int = 40,
    display_results: bool = False,
    include_df_output: bool = False
) -> pd.DataFrame:
    """
    Applies PCA on training and testing data and returns principal components.

    Args:
        X_train (pd.DataFrame): Training set features.
        X_test (pd.DataFrame): Testing set features.
        index (pd.Index): Index to apply to the resulting DataFrame.
        num_pca_components (int): Number of PCA components to keep.
        tuning_mode (bool): Print additional information if True (default True).

    Returns:
        pd.DataFrame: DataFrame containing principal components.
    """
    pca = PCA(n_components=num_pca_components)
    pca.fit(X_train)

    principal_components_train = pca.transform(X_train)
    principal_components_test = pca.transform(X_test)

    pca_columns = [f'pca{i}' for i in range(1, num_pca_components + 1)]
    principal_components_df = pd.DataFrame(
        np.concatenate([principal_components_train, principal_components_test]),
        columns=pca_columns,
        index=idx
    )

    if display_results:
        explained_variance = pca.explained_variance_ratio_.sum() * 100
        print(f"Explained variance of first {num_pca_components} components: {explained_variance:.2f}%")
        print("Explained variance ratio per component: ")
        components = np.arange(len(pca.explained_variance_ratio_))
        plt.bar(components,pca.explained_variance_ratio_)
        plt.title("Explained variance ratio per component")
        plt.ylabel("Proportioon of explanation")
        plt.xlabel("Principal Component")
        plt.show()
        print(f"Principal components shape: {principal_components_df.shape}")
    if include_df_output:
        print(principal_components_df.head())

    return principal_components_df


def create_pca_lag(
    principal_components_df: pd.DataFrame,
    shift_amount: int,
    num_lag_components: int = 5,
    display_results: bool = False
) -> pd.DataFrame:
    """
    Creates lagged principal components, meaning, it takes the principal
    components of previous days.

    Args:
        principal_components_df (pd.DataFrame): DataFrame of principal components.
        shift_amount (int): Number of positions to shift.
        num_lag_components (int): Number of lagged components to include (default 5).
        display_results (bool): Print additional information if True (default True).

    Returns:
        pd.DataFrame: DataFrame containing lagged principal components.
    """
    X_pc_lag = principal_components_df.iloc[:, :num_lag_components].shift(shift_amount)
    X_pc_lag.columns = [
        f'pca{i}_lag{shift_amount}' for i in range(1, num_lag_components + 1)
    ]

    if display_results:
        print(f"Lagged components shape {shift_amount}: {X_pc_lag.shape}")
        print(X_pc_lag.head(7))

    return X_pc_lag


def get_lagged_pca(principal_components_train_test_df, num_lags_to_include, display_results=False):
    """
    Generates lagged principal components in order
    to consider historical movements in the prediction.
    """
    X_pca_lag_list = []

    for shift in range(1, num_lags_to_include + 1):
        X_pca_lag = create_pca_lag(
            principal_components_train_test_df, 
            shift_amount=shift, 
            num_lag_components=num_lags_to_include, 
            display_results=display_results
        )
        X_pca_lag_list.append(X_pca_lag)

    # concatenate all lagged components into a single DataFrame
    X_pca_lags = pd.concat(X_pca_lag_list, axis=1)
    
    if display_results:
        print(f"\nShape of all pca combined: {X_pca_lags.shape}")
        print("X_pca_lags all",X_pca_lags.head())
    print(f"\n1-{num_lags_to_include} day-lag components included")
    print(f"Each lagged component considering {principal_components_train_test_df.shape[1]} principal components")
        
    return X_pca_lags


def combine_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    idx: pd.Index
) -> pd.DataFrame:
    """
    Combines training and testing datasets into a single DataFrame.

    Args:
        X_train (pd.DataFrame): Training data DataFrame.
        X_test (pd.DataFrame): Testing data DataFrame.
        index (pd.Index): Index for the combined DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame with the given index.
    """
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined.index = idx  # Ensure the index is set to the provided index
    
    return X_combined


def concatenate_with_pca_lags(
    X_raw: pd.DataFrame,
    X_pc_lags: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenates raw features with lagged principal components,
    and drop missing values.

    Args:
        X_raw (pd.DataFrame): Combined train and test features (excluding PCA).
        X_pc_lags (pd.DataFrame): Lagged principal components.
        tuning_mode (bool): Print additional information if True (default True).

    Returns:
        pd.DataFrame: Concatenated DataFrame without missing values.
    """
    X_combined = pd.concat([X_raw, X_pc_lags], axis=1).dropna()

    return X_combined

def concat_pca_to_X(
    X: pd.DataFrame,
    X_pca_lag: pd.DataFrame,
    display_results=False
):
    """Add the pca data to the features matrix,
    and delete initial missing values of the pca.
    Args:
        X (pd.DataFrame): Training data DataFrame.
        X_pc_lags (pd.DataFrame): Lagged principal components.
    """
    # Combines training and testing datasets into a single DataFrame.
    
    X_pca = pd.concat([X, X_pca_lag], axis=1).dropna()
    if display_results:
        print("initial rows: ", X.shape[0])
        print("final rows: ", X_pca.shape[0])
        print("Rows removed: ", X.shape[0] - X_pca.shape[0])
    return X_pca


def add_pca(X: pd.DataFrame, y: pd.DataFrame, 
    training_period_months: int, t: int = 5, n_components: int = 6,
    scaler_type: str = 'MinMaxScaler', display_results: bool = False,
    include_df_output: bool = False
):
    """
    Add t days of n principal components to the data.
    """ 
    X_train, _ , X_test, _ = split_training_test_data(
        X, y, training_period_months=training_period_months, 
        display_results=display_results
    )
    # Scaling of the data
    X_train_scaled, X_test_scaled = scale(
        X_train, X_test, scaler_type=scaler_type, display_results=display_results)
    
    # Calculation of Principal Components
    principal_components_train_test_df = adaboost_pca(
        X_train_scaled, 
        X_test_scaled, 
        idx=X.index, 
        num_pca_components=n_components, 
        display_results=display_results
    )
    # Getting historical movements of the data:
    # Generation of lagged principal components based on training data.
    X_pca_lag = get_lagged_pca(
        principal_components_train_test_df,
        num_lags_to_include=t,
        display_results=include_df_output
    )
    X_pc = concat_pca_to_X(
        X=X,
        X_pca_lag=X_pca_lag,
        display_results=include_df_output
    )
    assert X.shape[0] - X_pc.shape[0] == t, f"The number of rows removed is different than the lags: {t}"
    y_pc = y.loc[X_pc.index]
    
    # Extraction of last day (today) data to predict next day (tomoorrow)
    prediction_vector = pd.DataFrame(X_pc.iloc[-1,:].copy()).T
    X_pc = X_pc.iloc[:-1,:]
    
    y_pc = y_pc[:-1]
    print("""
        The lagged principal components have been succesfully added to the features matrix X, 
        the y component was trunc to match X's index,
        and the prediction vector was extracted for tomorrow signal.
    """
    )
    return X_pc, y_pc, prediction_vector


def random_over_sample(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    random_state: int = 19,
    display_results: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies random oversampling to balance the target classes in the training data.

    Args:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        tuning_mode (bool): Print additional information if True (default True).

    Returns:
        Tuple: Resampled X_train and y_train.
    """
    
    ros = RandomOverSampler(random_state=random_state)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    if display_results == True:
        print(f"y_train_resampled value_counts: \n{y_train_resampled.value_counts()}")
        
    return X_train_resampled, y_train_resampled
