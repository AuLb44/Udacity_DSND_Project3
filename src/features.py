"""
Feature engineering module with custom transformers and preprocessor building.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean and preprocess text data.
    """

    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)."""
        return self

    def transform(self, X):
        """
        Transform text data by lowercasing and stripping whitespace.

        Parameters
        ----------
        X : array-like
            Input text data.

        Returns
        -------
        np.ndarray
            Cleaned text data.
        """
        if hasattr(X, 'values'):
            X = X.values
        if X.ndim > 1:
            X = X.ravel()
        return np.array([
            str(text).lower().strip() if pd.notna(text) else ""
            for text in X
        ])


class TextCombiner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to combine multiple text columns into one.
    """

    def __init__(self, separator: str = " "):
        """
        Initialize the TextCombiner.

        Parameters
        ----------
        separator : str
            String to use when joining text columns.
        """
        self.separator = separator

    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)."""
        return self

    def transform(self, X):
        """
        Combine text columns into a single column.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Input data with text columns.

        Returns
        -------
        np.ndarray
            Combined text data.
        """
        if isinstance(X, pd.DataFrame):
            combined = X.apply(
                lambda row: self.separator.join(
                    str(val) if pd.notna(val) else "" for val in row
                ),
                axis=1
            )
            return combined.values
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X
            return np.array([
                self.separator.join(
                    str(val) if pd.notna(val) else "" for val in row
                )
                for row in X
            ])
        else:
            return np.array(X)


def build_numeric_transformer():
    """
    Build a pipeline for numeric features.

    Returns
    -------
    Pipeline
        Sklearn pipeline for numeric feature processing.
    """
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def build_categorical_transformer():
    """
    Build a pipeline for categorical features.

    Returns
    -------
    Pipeline
        Sklearn pipeline for categorical feature processing.
    """
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])


def build_text_transformer(max_features: int = 5000):
    """
    Build a pipeline for text features.

    Parameters
    ----------
    max_features : int
        Maximum number of features for TF-IDF vectorizer.

    Returns
    -------
    Pipeline
        Sklearn pipeline for text feature processing.
    """
    return Pipeline(steps=[
        ('combiner', TextCombiner()),
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english'))
    ])


def build_preprocessor(
    numeric_features: list,
    categorical_features: list,
    text_features: list,
    max_tfidf_features: int = 5000
):
    """
    Build a ColumnTransformer that handles numeric, categorical, and text columns.

    Parameters
    ----------
    numeric_features : list
        List of numeric column names.
    categorical_features : list
        List of categorical column names.
    text_features : list
        List of text column names.
    max_tfidf_features : int
        Maximum number of TF-IDF features.

    Returns
    -------
    ColumnTransformer
        Sklearn ColumnTransformer for preprocessing.
    """
    transformers = []

    if numeric_features:
        transformers.append((
            'numeric',
            build_numeric_transformer(),
            numeric_features
        ))

    if categorical_features:
        transformers.append((
            'categorical',
            build_categorical_transformer(),
            categorical_features
        ))

    if text_features:
        transformers.append((
            'text',
            build_text_transformer(max_features=max_tfidf_features),
            text_features
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )
