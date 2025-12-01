# Fashion Forward Forecasting - ML Pipeline Project

A ready-to-run machine learning pipeline for predicting product recommendations based on customer reviews. This project demonstrates text classification using scikit-learn with proper feature engineering and hyperparameter tuning.

## 🚀 Quick Start (Ready-to-Run)

This repository includes a sample dataset so you can test the entire pipeline immediately!

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd Udacity_DSND_Project3

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook

The easiest way to explore the pipeline:

```bash
cd notebooks
jupyter notebook 02_model_pipeline.ipynb
```

The notebook uses the included sample dataset (`data/raw/reviews_sample.csv`) and will:
- Load and preprocess the data
- Build the feature engineering pipeline
- Train with hyperparameter tuning (RandomizedSearchCV)
- Evaluate on a test set
- Save the model to `models/model_v1.joblib`

### 3. Train from Command Line (Alternative)

You can also train the model using Python directly:

```python
import sys
sys.path.insert(0, '.')

from src.data_processing import load_data, split_data, create_sample_data
from src.features import build_preprocessor
from src.model import build_model_pipeline, train_pipeline

# Create sample data (if needed)
create_sample_data('data/raw/reviews_sample.csv')

# Load data
df = load_data('data/raw/reviews_sample.csv')

# Split data
X_train, X_test, y_train, y_test = split_data(df, target_col='recommended')

# Build pipeline
preprocessor = build_preprocessor(
    numeric_cols=['age'],
    categorical_cols=['product_category'],
    text_col='review_text'
)
pipeline = build_model_pipeline(preprocessor)

# Train with hyperparameter tuning
search = train_pipeline(
    pipeline, X_train, y_train,
    n_iter=10, cv=3,
    save_path='models/model_v1.joblib'
)

print(f"Best F1 Score: {search.best_score_:.4f}")
print(f"Best Parameters: {search.best_params_}")
```

## 📁 Project Structure

```
├── data/
│   └── raw/
│       └── reviews_sample.csv    # Sample dataset (15 rows) for testing
├── models/
│   └── model_v1.joblib           # Saved trained model (after training)
├── notebooks/
│   └── 02_model_pipeline.ipynb   # Complete pipeline notebook
├── src/
│   ├── __init__.py               # Package initialization
│   ├── data_processing.py        # Data loading and preprocessing
│   ├── features.py               # Feature engineering transformers
│   └── model.py                  # Model training and tuning
├── starter/
│   └── ...                       # Original starter files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📊 Data Format

The sample dataset (`data/raw/reviews_sample.csv`) contains:
- `review_text`: Customer review text
- `age`: Customer age
- `product_category`: Product category (Dresses, Tops, etc.)
- `recommended`: Target variable (1 = recommended, 0 = not recommended)

For the full dataset, use `starter/data/reviews.csv` with columns:
- `Clothing ID`, `Age`, `Title`, `Review Text`, `Positive Feedback Count`
- `Division Name`, `Department Name`, `Class Name`
- `Recommended IND` (target)

## ⚙️ Pipeline Features

### Feature Engineering
- **Numeric features**: Median imputation + StandardScaler
- **Categorical features**: Constant imputation + OneHotEncoder
- **Text features**: Text cleaning + TF-IDF vectorization

### Hyperparameter Tuning
- Uses `RandomizedSearchCV` with `StratifiedKFold` cross-validation
- Default search space includes:
  - TF-IDF: `max_features` (2000-10000), `ngram_range` ((1,1), (1,2))
  - Classifier: `C` (0.01-10), `penalty` (l2), `solver` (lbfgs, saga)

### Model
- Logistic Regression with max_iter=2000 for convergence
- Optimized for F1 score by default

## 🧪 Testing the Pipeline

Run the notebook with the sample data to verify everything works:

```bash
cd notebooks
jupyter notebook 02_model_pipeline.ipynb
# Run all cells
```

Expected output:
- Model trains successfully with RandomizedSearchCV
- Evaluation metrics displayed (accuracy, precision, recall, F1, ROC-AUC)
- Model saved to `models/model_v1.joblib`
- Inference demo on sample inputs

## 📦 Dependencies

See `requirements.txt` for the complete list. Key packages:
- pandas >= 1.3.0
- scikit-learn >= 1.0, < 2.0
- joblib >= 1.0.0
- matplotlib, seaborn (for visualization)
- jupyter (for running notebooks)

## 🔧 Configuration

All random seeds are set for reproducibility:
- `random_state=42` for train/test split
- `random_state=42` for RandomizedSearchCV
- `random_state=42` for LogisticRegression

## License

[License](LICENSE.txt)
