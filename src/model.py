"""
Model building and training module.
"""

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from .features import build_preprocessor
from .data_processing import get_column_types


def build_pipeline(
    numeric_features: list,
    categorical_features: list,
    text_features: list,
    max_tfidf_features: int = 5000,
    random_state: int = 27
):
    """
    Build a complete sklearn Pipeline with preprocessing and classifier.

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
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Complete sklearn Pipeline.
    """
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        text_features=text_features,
        max_tfidf_features=max_tfidf_features
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs'
        ))
    ])

    return pipeline


def get_param_grid():
    """
    Get parameter grid for hyperparameter search.

    Returns
    -------
    dict
        Parameter grid for RandomizedSearchCV.
    """
    return {
        'preprocessor__text__tfidf__max_features': [1000, 3000, 5000],
        'preprocessor__text__tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': np.logspace(-3, 3, 7),
        'classifier__penalty': ['l2'],
        'classifier__class_weight': [None, 'balanced']
    }


def train_pipeline(
    pipeline: Pipeline,
    X_train,
    y_train,
    param_grid: dict = None,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = 'f1',
    random_state: int = 27,
    n_jobs: int = -1,
    verbose: int = 1
):
    """
    Train pipeline using RandomizedSearchCV.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn Pipeline to train.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    param_grid : dict, optional
        Parameter grid for search. If None, uses default.
    n_iter : int
        Number of parameter settings sampled.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs.
    verbose : int
        Verbosity level.

    Returns
    -------
    RandomizedSearchCV
        Fitted RandomizedSearchCV object.
    """
    if param_grid is None:
        param_grid = get_param_grid()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    return search


def save_pipeline(pipeline, filepath: str = 'models/model_v1.joblib'):
    """
    Save a trained pipeline to disk.

    Parameters
    ----------
    pipeline : Pipeline or RandomizedSearchCV
        Trained pipeline or search object to save.
    filepath : str
        Path to save the pipeline.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the best estimator if it's a search object
    if hasattr(pipeline, 'best_estimator_'):
        joblib.dump(pipeline.best_estimator_, filepath)
    else:
        joblib.dump(pipeline, filepath)

    print(f"Pipeline saved to {filepath}")


def load_pipeline(filepath: str = 'models/model_v1.joblib'):
    """
    Load a trained pipeline from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved pipeline.

    Returns
    -------
    Pipeline
        Loaded pipeline.
    """
    return joblib.load(filepath)
