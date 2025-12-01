"""Feature engineering utilities for Fashion Forward Forecasting."""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


class TextCleaner(BaseEstimator, TransformerMixin):
    """Transformer for cleaning text data.
    
    Performs basic text preprocessing:
    - Lowercase conversion
    - Remove special characters
    - Remove extra whitespace
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """Clean the text data.
        
        Args:
            X: Array-like of text strings.
            
        Returns:
            Array of cleaned text strings.
        """
        # Ensure we're working with a 1D array of strings
        if hasattr(X, 'values'):
            X = X.values
        if len(X.shape) > 1:
            X = X.ravel()
        
        cleaned = []
        for text in X:
            if not isinstance(text, str):
                text = str(text) if text is not None else ''
            # Lowercase
            text = text.lower()
            # Remove special characters but keep spaces and basic punctuation
            text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            cleaned.append(text)
        
        return np.array(cleaned)


class ExtraTextFeatures(BaseEstimator, TransformerMixin):
    """Transformer for extracting additional text features.
    
    Extracts:
    - Review length (character count)
    - Word count
    - Exclamation count
    - Simple sentiment score (using word lists)
    """
    
    def __init__(self):
        # Simple positive/negative word lists for basic sentiment
        self.positive_words = {
            'love', 'great', 'excellent', 'amazing', 'wonderful', 'perfect',
            'best', 'beautiful', 'fantastic', 'gorgeous', 'awesome', 'lovely',
            'favorite', 'recommend', 'comfortable', 'flattering', 'quality'
        }
        self.negative_words = {
            'hate', 'terrible', 'awful', 'horrible', 'worst', 'bad', 'poor',
            'disappointed', 'disappointing', 'cheap', 'uncomfortable', 'ugly',
            'return', 'returned', 'waste', 'regret', 'avoid'
        }
    
    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """Extract text features.
        
        Args:
            X: Array-like of text strings.
            
        Returns:
            2D array with columns [length, word_count, exclamation_count, sentiment].
        """
        if hasattr(X, 'values'):
            X = X.values
        if len(X.shape) > 1:
            X = X.ravel()
        
        features = []
        for text in X:
            if not isinstance(text, str):
                text = str(text) if text is not None else ''
            
            text_lower = text.lower()
            words = text_lower.split()
            
            # Features
            length = len(text)
            word_count = len(words)
            exclamation_count = text.count('!')
            
            # Simple sentiment score
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            sentiment = positive_count - negative_count
            
            features.append([length, word_count, exclamation_count, sentiment])
        
        return np.array(features)


def build_preprocessor(
    numeric_cols: list,
    categorical_cols: list,
    text_col: str
) -> ColumnTransformer:
    """Build a ColumnTransformer for preprocessing features.
    
    Creates a preprocessor with:
    - Numeric pipeline: SimpleImputer(median) -> StandardScaler
    - Categorical pipeline: SimpleImputer(constant) -> OneHotEncoder
    - Text pipeline: TextCleaner -> TfidfVectorizer
    
    Args:
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.
        text_col: Name of the text column.
        
    Returns:
        ColumnTransformer with configured pipelines.
    """
    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Text pipeline - TextCleaner outputs cleaned strings, TfidfVectorizer converts to features
    text_pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 1)))
    ])
    
    # Combine all pipelines
    transformers = []
    
    if numeric_cols:
        transformers.append(('num', numeric_pipeline, numeric_cols))
    
    if categorical_cols:
        transformers.append(('cat', categorical_pipeline, categorical_cols))
    
    if text_col:
        transformers.append(('txt', text_pipeline, text_col))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return preprocessor
