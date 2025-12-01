"""
Data processing module for loading and preparing data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_csv(filepath)


def prepare_features_target(
    df: pd.DataFrame,
    target_column: str = 'Recommended IND'
):
    """
    Separate features and target from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and target.
    target_column : str
        Name of the target column.

    Returns
    -------
    tuple
        (X, y) where X is the features DataFrame and y is the target Series.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.1,
    random_state: int = 27
):
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame.
    y : pd.Series
        Target Series.
    test_size : float
        Proportion of dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state
    )


def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercase and strip whitespace.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    str
        Cleaned text string.
    """
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def get_column_types(df: pd.DataFrame, target_column: str = 'Recommended IND'):
    """
    Identify numeric, categorical, and text columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str
        Name of the target column to exclude.

    Returns
    -------
    dict
        Dictionary with keys 'numeric', 'categorical', 'text' containing
        lists of column names.
    """
    # Define text columns (known from the dataset)
    text_cols = ['Title', 'Review Text']

    # Define categorical columns
    cat_cols = ['Division Name', 'Department Name', 'Class Name']

    # Define numeric columns
    numeric_cols = ['Clothing ID', 'Age', 'Positive Feedback Count']

    # Filter to only include columns that exist in the DataFrame
    features = df.drop(target_column, axis=1, errors='ignore')

    return {
        'numeric': [col for col in numeric_cols if col in features.columns],
        'categorical': [col for col in cat_cols if col in features.columns],
        'text': [col for col in text_cols if col in features.columns]
    }
