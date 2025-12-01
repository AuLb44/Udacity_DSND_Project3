"""Model training utilities for Fashion Forward Forecasting."""

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def build_model_pipeline(preprocessor) -> Pipeline:
    """Build the model pipeline with preprocessor and classifier.
    
    Args:
        preprocessor: A fitted or unfitted ColumnTransformer preprocessor.
        
    Returns:
        sklearn Pipeline with preprocessor and LogisticRegression classifier.
    """
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, random_state=42))
    ])
    
    return pipeline


def train_pipeline(
    pipeline: Pipeline,
    X_train,
    y_train,
    param_distributions: dict = None,
    n_iter: int = 50,
    cv: int = 5,
    scoring: str = 'f1',
    random_state: int = 42,
    save_path: str = 'models/model_v1.joblib',
    verbose: int = 1
):
    """Train the pipeline using RandomizedSearchCV for hyperparameter tuning.
    
    Uses StratifiedKFold cross-validation to avoid data leakage and ensure
    proper evaluation across class distributions.
    
    Args:
        pipeline: sklearn Pipeline to train.
        X_train: Training features.
        y_train: Training labels.
        param_distributions: Dict of hyperparameters to search. If None, uses defaults.
        n_iter: Number of parameter settings to sample.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for evaluation.
        random_state: Random seed for reproducibility.
        save_path: Path to save the best model. Set to None to skip saving.
        verbose: Verbosity level for RandomizedSearchCV.
        
    Returns:
        Fitted RandomizedSearchCV object.
    """
    # Default parameter distributions for hyperparameter tuning
    if param_distributions is None:
        param_distributions = {
            # TF-IDF parameters
            'preproc__txt__tfidf__max_features': [2000, 5000, 10000],
            'preproc__txt__tfidf__ngram_range': [(1, 1), (1, 2)],
            # Classifier parameters
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs', 'saga'],
        }
    
    # Use StratifiedKFold for proper cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Create RandomizedSearchCV
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=verbose,
        return_train_score=True
    )
    
    # Fit the search
    search.fit(X_train, y_train)
    
    # Save the best model if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(search.best_estimator_, save_path)
        print(f"Best model saved to: {save_path}")
    
    return search
