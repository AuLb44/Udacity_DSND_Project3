"""Data processing utilities for Fashion Forward Forecasting."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file.
    
    Args:
        path: Path to the CSV file.
        
    Returns:
        DataFrame containing the loaded data.
    """
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame,
    target_col: str = 'recommended',
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """Split data into train and test sets.
    
    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col].copy()
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def normalize_text_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Normalize text column by lowercasing and stripping whitespace.
    
    Args:
        df: Input DataFrame.
        col: Name of the text column to normalize.
        
    Returns:
        DataFrame with normalized text column.
    """
    df = df.copy()
    df[col] = df[col].astype(str).str.lower().str.strip()
    return df


def create_sample_data(path: str = 'data/raw/reviews_sample.csv') -> str:
    """Create a small sample CSV dataset for testing.
    
    Creates a CSV file with 15 example rows containing columns:
    review_text, age, product_category, recommended
    
    Args:
        path: Path to save the sample CSV file.
        
    Returns:
        Path to the created file.
    """
    sample_data = {
        'review_text': [
            "Absolutely love this dress! The fit is perfect and the color is exactly as shown. Great quality fabric!",
            "Not worth the price. The material feels cheap and the sizing runs very small. Disappointed.",
            "This top is cute but runs a bit large. I should have sized down. Still keeping it though.",
            "Perfect for summer! Lightweight and comfortable. I've received so many compliments!",
            "The stitching came undone after one wash. Poor quality. Would not recommend.",
            "Beautiful blouse! The embroidery detail is gorgeous. Runs true to size.",
            "Meh. It's okay. Nothing special but decent for the price.",
            "This is now my favorite sweater! So soft and cozy. Worth every penny!",
            "Too sheer - you definitely need to wear something underneath. Otherwise nice design.",
            "Love love love! The pattern is unique and the quality is excellent!",
            "Returned immediately. The color was completely different from the photo.",
            "Great basic tee. Nothing fancy but good quality and comfortable fit.",
            "This jacket exceeded my expectations! The hardware is nice and it fits perfectly.",
            "The dress arrived wrinkled and the zipper was stuck. Very frustrated with this purchase.",
            "Finally found jeans that fit my petite frame! Will definitely buy more colors."
        ],
        'age': [34, 45, 28, 52, 31, 67, 23, 41, 36, 29, 55, 38, 44, 26, 33],
        'product_category': [
            'Dresses', 'Tops', 'Tops', 'Dresses', 'Blouses',
            'Blouses', 'Tops', 'Sweaters', 'Tops', 'Dresses',
            'Dresses', 'Tops', 'Jackets', 'Dresses', 'Pants'
        ],
        'recommended': [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_csv(path, index=False)
    
    return path
