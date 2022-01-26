# Vixm Principal Compenents and Training Functions

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import datetime
from pandas.tseries.offsets import DateOffset





def split_training_test_data(X, y, training_period_months=12, tuning_mode=True):
    """
    This function split the preprocessed data of a time series into two windows: a training window and a testing window,
    Its give back the features and signals divded accordingly
    
    Args:
    X: a pandas dataframe with the features in its columns, using a datetime index
    y: a pandas dataframe with the signal, using a datetime index
    training_period_months: number of months for the training period of the data counting from the beggining of the data
    tuning_mode: a boolean parameter set for cases when tuning is made. Automatically set to True unless provided a False.
    
    Return:
    Two pandas dataframes and two series in the following order:
    X_train: a pandas dataframe with the features of the train window
    y_train: a pandas series with the signals of the train window 
    X_test: a pandas dataframe with the featires in the test window
    y_test: a pandas series with the signals of the train window 
    
    """

    training_begin = X.index.min()
    training_end = X.index.min() + DateOffset(months=training_period_months)
    

    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]

    X_test = X.loc[training_end + DateOffset(days=1):]
    y_test = y.loc[training_end + DateOffset(days=1):]

    if tuning_mode == True:
        print(f"Training dates: {training_begin} to {training_end}")
        print("y_train value_counts:")
        display(y_train.value_counts())
        print(f"y_test shape: {y_test.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_train tail(1):")
        display(X_train.tail(1))
        print(f"X_train head(1): ")
        display(X_test.head(1))
    return X_train, y_train, X_test, y_test



def scale(X_train, X_test, type='MinMaxScaler', tuning_mode= True):
    """
    This function apply Min Max or Standard scaling to a divided set of features divided as train and test data
    
    Args:
    The two dataframes:
    X_train: a pandas dataframe with features of the training window
    X_test:  a pandas dataframe with features of the test window
    tuning_mode: a boolean parameter set for cases when tuning is made. Automatically set to True unless provided a False.
    
    Return:
    Two arrays coming from the original dataframes after applying StandardScaler() or MinMaxScaler(), where the standarization is made using the X_train features
    """
    # Create an Scaler instance
    scaler =  MinMaxScaler() 
    
    if type=='StandardScaler':
        scaler =  StandardScaler()  
 
    # Apply the scaler model to fit the X_train data
    X_scaler = scaler.fit(X_train)

    # Transform the X_train and X_test DataFrames using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    if tuning_mode == True:
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled



def adaboost_pca(X_train, X_test, index, num_pca_components=40,  tuning_mode= True):
    
    """
    This function calculates the principal components of an X features matrix, already divided in a train and a test set.
    
    Args:
    X_train: a pandas dataframe with the training set of features
    X_test: a pandas dataframe with the test set of features
    tuning_mode: a boolean parameter set for cases when tuning is made. Automatically set to True unless provided a False.
    
    Returns:
    An X pandas dataframe of features, with the train and test samples concatenated, which correspond
    to the principal components of the original data, calculated using the transformation calculated from the train set of data.
    """
    
    # Initiate and calculate principal components transformation based on the train data
    pca = PCA(n_components = num_pca_components)
    pca.fit(X_train)
    
    # Calculate train and test principal components using the trained model
    principal_components_train = pca.transform(X_train)
    principal_components_test  = pca.transform(X_test)
    
    # Name principal components columns properly
    pca_column_list = []
    for i in range(1, num_pca_components+1):
        pca_column_list.append(f"pca{i}")

    #Concatenate train and test principal components in one dataframe called principal_components_train_test_df
    principal_components_train_test = np.concatenate((principal_components_train, principal_components_test), axis = 0)
    principal_components_train_test_df = pd.DataFrame(data = principal_components_train_test, columns = pca_column_list, index = index)
    if tuning_mode == True:
        explained_variance=sum(pca.explained_variance_ratio_)
        print(f"Principal Components explained variance: {explained_variance*100:.2f}%")
        print(f"Principal Components Shape:              {principal_components_train_test_df.shape}")
        print("principal_components_train_test_df head(5):")
        display(principal_components_train_test_df.head(5))
    return principal_components_train_test_df





def create_pca_lag(principal_components_train_test_df, shift_amount, number_of_pca_lag_component_to_include=5, tuning_mode=True ):
    """
    This function creates a new dataframe by taking the first 3 first principal components (in the first three columns) and shifting them in a desire number
    
    Args:
    principal_components_train_test_df: a pandas dataframe 
    shift_amount: the number of positions to shift the first three columnms
    tuning_mode: a boolean parameter set for cases when tuning is made. Automatically set to True unless provided a False.
    
    Return:
    A pandas dataframe containing the shifted 3 first columns of the dataframe,
    with column names 'pca1_lag1','pca2_lag1','pca3_lag1'
    """
    X_pc_lag = principal_components_train_test_df.iloc[:,0:(number_of_pca_lag_component_to_include-1)]
    if tuning_mode == True:
        print(f"X_pc_lag shape: {X_pc_lag.shape}")

    X_pc_lag.columns = ['pca1_lag1','pca2_lag1','pca3_lag1']
    X_pc_lag = X_pc_lag.shift(shift_amount)

    if tuning_mode == True:
        print(f"X_pc_lag: {X_pc_lag}")
        print(f"X_pc_lag shape. {X_pc_lag.shape}")
    return X_pc_lag




def concatenate_lags(X_pc_lag1, X_pc_lag2, X_pc_lag3, X_pc_lag4, X_pc_lag5, tuning_mode = True):
    """
    This function concatenates the 5 principal components features in one dataframe, by column.
    
    Arguments:
    X_pc_lag1,...X_pc_lag5: are five dataframes, in our case containing principal components at different lags (1 to 5 days)
    tuning_mode: a boolean parameter set for cases when tuning is made. Automatically set to True unless provided a False.
    """ 
    
    X_pc_lags=pd.concat([X_pc_lag1, 
                         X_pc_lag2, 
                         X_pc_lag3, 
                         X_pc_lag4, 
                         X_pc_lag5], 
                         axis=1
    )
    
    if tuning_mode == True:
        print(f"X_pc_lags shape: {X_pc_lags.shape}")
    return X_pc_lags



def combine_train_test(X_train, X_test, index):
    """
    This function concantenate the train and test arrays, and apply the proper index, to get back to an X dataframe
    
    Args:
    X_train, X_test: arrays to concatenate, wich should have the same number of columns
    
    Return:
    The X dataframe as a pandas dataframe, and the index as the last X dataframe of features
    """
    
    X_combined = np.concatenate([X_train, X_test], axis = 0)
    X_combined = pd.DataFrame(data = X_combined, index=index)
    return X_combined




def concatenate_with_pca_lags(X_raw, X_pc_lags, tuning_mode = True):
    """
    This function concatenates all the sources of data: features and lags principal compoents.
    It also eliminates NaNs due to lag
    
    Args:
    X_raw: the combination of X_train and X_test features, excluding principal compoentns
    X_pc_lags: the 3 main principal components of the set of features, lagged in 1, 2, 3, 4 and 5 days.
    tuning_mode: a boolean parameter set for cases when tuning is made. Automatically set to True unless provided a False.
    
    Returns:
    A dataframe containing the concatenation of all features and principal components without missing values, with the proper datetime index
    """
    
    X_pc = pd.concat([X_raw, X_pc_lags], axis=1)

    if tuning_mode == True:
        print(f"X_pc shape: {X_pc.shape}" )
    return X_pc




def eliminate_nans_in_pca_data(X_pc, y, n=5, tuning_mode = True):
    """
    This function eliminates the first n missing values of a dataframe.
    
    Args:
    X_pc, y: two dataframes
    n: number of initial rows to delete
    tuning_mode: a boolean parameter set for cases when tuning is made. Automatically set to True unless provided a False.
    
    Return:
    The two dataset without the first n rows of each, in the same order that were given.
    """
    X_pc = X_pc[n:-1]
    y = y[n:-1]

    if tuning_mode == True:
        print(f"X_pc.shape: {X_pc.shape}")
        print(f"y shape: {y.shape}")
    return X_pc, y



def random_over_sample(X_train, y_train, tuning_mode=True):
    """
    This function uses RandomOverSampler to resample the X and Y using random_state=1.
    Per default, this applies to training data. The function is used to get a balanced number of signals 0 & 1 for y_train.
    Mostly used with unbalanced data.
    
    Args:
    X_train: array-like, dataframe, sparse matrix of shape (n_samples, n_features). Data array.
    y_train: array-like of shape (n_samples,). Target array.
    
    
    Return:
    The X and y datasets resamples, meaning, with the same amount of observations for each different element of y.
    
    For more details check:
    https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
    
    """
    
    ros = RandomOverSampler(random_state = 1)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    if tuning_mode == True:
        print(f"y_train_resampled value_counts: {y_train_resampled.value_counts()}")
    return X_train_resampled, y_train_resampled






