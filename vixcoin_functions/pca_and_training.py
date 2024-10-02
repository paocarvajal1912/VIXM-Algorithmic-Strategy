
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


def split_training_test_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    training_period_months: int = 12,
    tuning_mode: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits a time-series dataset into training and testing windows.

    Args:
        X (pd.DataFrame): DataFrame containing features with a datetime index.
        y (pd.DataFrame): DataFrame or Series containing target signals.
        training_period_months (int): Number of months for the training period.
        tuning_mode (bool): Print additional information if True (default True).

    Returns:
        Tuple: X_train, y_train, X_test, y_test.
    """
    training_begin = pd.to_datetime(X.index.min())
    training_end = training_begin + DateOffset(months=training_period_months)
    print(X.head())

    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]
    X_test = X.loc[training_end + DateOffset(days=1):]
    y_test = y.loc[training_end + DateOffset(days=1):]

    if tuning_mode:
        print(f"Training period: {training_begin} to {training_end}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print("y_train value counts:")
        print(y_train.value_counts())

    return X_train, y_train, X_test, y_test


def scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = 'MinMaxScaler',
    tuning_mode: bool = True
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

    if tuning_mode:
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled


def adaboost_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    idx: pd.Index,
    num_pca_components: int = 40,
    tuning_mode: bool = True
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

    if tuning_mode:
        explained_variance = pca.explained_variance_ratio_.sum() * 100
        print(f"Explained variance: {explained_variance:.2f}%")
        print(f"Principal components shape: {principal_components_df.shape}")
        print(principal_components_df.head())

    return principal_components_df


def create_pca_lag(
    principal_components_df: pd.DataFrame,
    shift_amount: int,
    num_lag_components: int = 5,
    tuning_mode: bool = True
) -> pd.DataFrame:
    """
    Creates lagged principal components.

    Args:
        principal_components_df (pd.DataFrame): DataFrame of principal components.
        shift_amount (int): Number of positions to shift.
        num_lag_components (int): Number of lagged components to include (default 5).
        tuning_mode (bool): Print additional information if True (default True).

    Returns:
        pd.DataFrame: DataFrame containing lagged principal components.
    """
    X_pc_lag = principal_components_df.iloc[:, :num_lag_components].shift(shift_amount)
    X_pc_lag.columns = [
        f'pca{i}_lag{shift_amount}' for i in range(1, num_lag_components + 1)
    ]

    if tuning_mode:
        print(f"Lagged components shape: {X_pc_lag.shape}")

    return X_pc_lag



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
    X_pc_lags: pd.DataFrame,
    tuning_mode: bool = True
) -> pd.DataFrame:
    """
    Concatenates raw features with lagged principal components.

    Args:
        X_raw (pd.DataFrame): Combined train and test features (excluding PCA).
        X_pc_lags (pd.DataFrame): Lagged principal components.
        tuning_mode (bool): Print additional information if True (default True).

    Returns:
        pd.DataFrame: Concatenated DataFrame without missing values.
    """
    X_combined = pd.concat([X_raw, X_pc_lags], axis=1).dropna()

    if tuning_mode:
        print(f"Final dataset shape: {X_combined.shape}")

    return X_combined


def eliminate_nans_in_pca_data(
    X_pc: pd.DataFrame,
    y: pd.Series,
    n: int = 5,
    tuning_mode: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Removes the first `n` rows of missing values from the feature and target data.

    Args:
        X_pc (pd.DataFrame): DataFrame containing PCA features.
        y (pd.Series): Target signal Series.
        n (int): Number of rows to remove (default is 5).
        tuning_mode (bool): Print additional information if True (default True).

    Returns:
        Tuple: Cleaned X and y data.
    """
    X_pc_cleaned = X_pc.iloc[n:]#.reset_index(drop=drop_index)
    y_cleaned = y.iloc[n:]#.reset_index(drop=drop_index)
    #y_cleaned.index = X_pc_cleaned.index

    if tuning_mode:
        print(f"Cleaned X shape: {X_pc_cleaned.shape}, Cleaned y shape: {y_cleaned.shape}")
        print(X_pc_cleaned.head())
        print(f"Cleaned y shape: {y.shape}, Cleaned y shape: {y.shape}")
        print(y.head())

    return X_pc_cleaned, y_cleaned


def random_over_sample(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    tuning_mode: bool = True
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
    
    ros = RandomOverSampler(random_state=1)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train,y_train)

    if tuning_mode == True:
        print(f"y_train_resampled value_counts: {y_train_resampled.value_counts()}")
        
    return X_train_resampled, y_train_resampled
