"""
Data preprocessing utilities for fall detection project.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Union


def subject_wise_split(
    subs: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits trial indices into training, validation, and testing sets based on unique subjects,
    ensuring that all trials from a given subject are allocated to the same data split.
    
    Args:
        subs: A pandas Series where each element represents the subject ID for a
              corresponding trial.
        test_size: The proportion of the dataset to include in the test split.
                   Should be between 0.0 and 1.0.
        val_size: The proportion of the training set to include in the validation
                  split after the initial train/test split. Should be between 0.0 and 1.0.
        random_state: Random seed for reproducibility.
        
    Returns:
        A tuple containing three numpy arrays:
        - train_indices: Indices of trials belonging to the training set.
        - val_indices: Indices of trials belonging to the validation set.
        - test_indices: Indices of trials belonging to the testing set.
    """
    # Get and split unique subjects
    subjects = sorted(subs.unique())
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=test_size,
        random_state=random_state
    )
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=val_size/(1-test_size),
        random_state=random_state
    )

    # Get indices for train, validation and test sets according to subs
    train_indices = subs[subs.isin(train_subjects)].index
    val_indices = subs[subs.isin(val_subjects)].index
    test_indices = subs[subs.isin(test_subjects)].index

    # Shuffle indices
    train_indices = np.random.permutation(train_indices)
    val_indices = np.random.permutation(val_indices)
    test_indices = np.random.permutation(test_indices)

    return train_indices, val_indices, test_indices


def normalize_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: Union[StandardScaler, MinMaxScaler, RobustScaler]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalizes the feature sets (train, validation, and test) using a given scaler,
    fitted on training data only to prevent data leakage.
    
    Args:
        X_train: The training feature set (pandas DataFrame).
        X_val: The validation feature set (pandas DataFrame).
        X_test: The test feature set (pandas DataFrame).
        scaler: a scikit-learn scaler (StandardScaler, MinMaxScaler)
                to be used for normalization.
                
    Returns:
        A tuple containing three pandas DataFrames:
        - The normalized training feature set.
        - The normalized validation feature set.
        - The normalized test feature set.
    """
    scaler.fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test)
    )