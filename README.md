# Fashion Forward Forecasting - Udacity DSND Project 3

This project implements an end-to-end machine learning pipeline for predicting whether a customer will recommend a product based on their review. The pipeline properly handles numeric, categorical, and text data using sklearn's `ColumnTransformer` and `Pipeline` classes.

## Project Layout

```
.
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── LICENSE.txt                   # License file
├── data/
│   └── raw/
│       └── reviews.csv           # Dataset (not included, add manually)
├── models/
│   └── model_v1.joblib           # Saved trained model (generated after training)
├── notebooks/
│   └── 02_model_pipeline.ipynb   # Example notebook demonstrating the pipeline
├── src/
│   ├── __init__.py               # Package initialization
│   ├── data_processing.py        # Data loading and preprocessing utilities
│   ├── features.py               # Custom transformers and feature engineering
│   └── model.py                  # Pipeline building and training functions
└── starter/
    ├── data/
    │   └── reviews.csv           # Starter dataset
    └── starter.ipynb             # Starter notebook
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Dependencies

The project requires the following Python packages:

```
scikit-learn
pandas
numpy
joblib
notebook
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Udacity_DSND_Project3
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare the dataset:
```bash
# Copy the reviews.csv file to the data/raw directory
mkdir -p data/raw
cp starter/data/reviews.csv data/raw/
```

## Running the Example

### Using Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/02_model_pipeline.ipynb`

3. Run all cells to:
   - Load and explore the data
   - Build the preprocessing pipeline
   - Train with hyperparameter search
   - Evaluate on the test set
   - Save the trained model

### Using Python Scripts

You can also use the src modules directly in your Python scripts:

```python
from src.data_processing import load_data, prepare_features_target, split_data, get_column_types
from src.model import build_pipeline, train_pipeline, save_pipeline

# Load and prepare data
df = load_data('data/raw/reviews.csv')
X, y = prepare_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Get column types
col_types = get_column_types(df)

# Build and train pipeline
pipeline = build_pipeline(
    numeric_features=col_types['numeric'],
    categorical_features=col_types['categorical'],
    text_features=col_types['text']
)

search = train_pipeline(pipeline, X_train, y_train)

# Save the model
save_pipeline(search, 'models/model_v1.joblib')
```

## Pipeline Features

### Data Processing (`src/data_processing.py`)
- `load_data()`: Load CSV data
- `prepare_features_target()`: Separate features and target
- `split_data()`: Train/test split with reproducibility
- `get_column_types()`: Identify numeric, categorical, and text columns

### Feature Engineering (`src/features.py`)
- `TextCleaner`: Custom transformer for text preprocessing
- `TextCombiner`: Combine multiple text columns
- `build_preprocessor()`: Create a `ColumnTransformer` that handles:
  - Numeric features: imputation + standard scaling
  - Categorical features: imputation + one-hot encoding
  - Text features: combine + clean + TF-IDF vectorization

### Model Training (`src/model.py`)
- `build_pipeline()`: Create sklearn Pipeline with preprocessor and classifier
- `train_pipeline()`: Train using RandomizedSearchCV
- `save_pipeline()` / `load_pipeline()`: Model persistence using joblib

## Dataset

The dataset includes the following features:

| Feature | Type | Description |
|---------|------|-------------|
| Clothing ID | Numeric | Specific piece being reviewed |
| Age | Numeric | Reviewer's age |
| Title | Text | Review title |
| Review Text | Text | Review body |
| Positive Feedback Count | Numeric | Number of positive feedbacks |
| Division Name | Categorical | Product high-level division |
| Department Name | Categorical | Product department |
| Class Name | Categorical | Product class |
| **Recommended IND** | **Target** | 1 = recommended, 0 = not recommended |

## Reproducibility

Random seeds are set throughout the code for reproducibility:
- Data splitting: `random_state=27`
- Model training: `random_state=27`
- Hyperparameter search: `random_state=27`

## Built With

* [scikit-learn](https://scikit-learn.org/) - Machine learning library
* [pandas](https://pandas.pydata.org/) - Data manipulation
* [numpy](https://numpy.org/) - Numerical computing
* [Jupyter](https://jupyter.org/) - Interactive notebooks

## License

[License](LICENSE.txt)
