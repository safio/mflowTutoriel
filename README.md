# MLflow Tutorial Project

A simple MLflow project for learning machine learning experiment tracking using the Iris flower classification dataset.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run a single experiment:
```bash
python train.py --n_estimators 100 --max_depth 10
```

### Run with MLflow project:
```bash
python -m mlflow run . -P n_estimators=200 -P max_depth=15 --env-manager local --experiment-name iris-classification
```

### View results in MLflow UI:
```bash
# Option 1: Using Python module (recommended)
python -m mlflow ui --host 0.0.0.0 --port 5001

# Option 2: Using the startup script
python start_mlflow.py
```

Then open http://localhost:5001 in your browser.

### Test your trained model:
```bash
# Test with built-in sample data
python test_model.py --use-sample

# Test with your own CSV file
python test_model.py --data-file sample_test_data.csv

# Test with a specific model run
python test_model.py --run-id YOUR_RUN_ID --use-sample

# Test and log results to MLflow UI
python test_model.py --use-sample --log-to-mlflow
python test_model.py --data-file sample_test_data.csv --log-to-mlflow
```

### Compare different ML algorithms:
```bash
# Compare all available algorithms (both scripts now use organized outputs)
python compare_models.py

# Compare specific models only
python compare_models.py --models SVM_Linear LogisticRegression KNN

# Use the advanced structured version with detailed logging
python compare_models_structured.py --models SVM_Linear LogisticRegression KNN

# Customize data split, random state, and output directory
python compare_models.py --test-size 0.3 --random-state 123 --output-dir my_outputs
python compare_models_structured.py --test-size 0.3 --random-state 123 --output-dir my_outputs
```

### Hyperparameter Tuning:
```bash
# Tune all models with both Grid and Random search (comprehensive but slow)
python hyperparameter_tuning.py

# Tune specific models only
python hyperparameter_tuning.py --models RandomForest SVM_Linear LogisticRegression

# Use only Random search (faster)
python hyperparameter_tuning.py --search-type random --n-iter 50

# Use only Grid search (thorough but slower)
python hyperparameter_tuning.py --search-type grid

# Quick tuning with fewer CV folds and iterations
python hyperparameter_tuning.py --models KNN DecisionTree --search-type random --n-iter 20 --cv-folds 3

# Custom settings
python hyperparameter_tuning.py --test-size 0.3 --random-state 123 --cv-folds 5
```

### Cross-Validation Analysis:
```bash
# Cross-validate all models with default strategies (KFold, StratifiedKFold)
python cross_validation.py

# Cross-validate specific models only
python cross_validation.py --models RandomForest SVM_Linear KNN

# Use specific CV strategies
python cross_validation.py --cv-strategies KFold StratifiedKFold ShuffleSplit

# Customize number of folds
python cross_validation.py --n-splits 10

# Quick analysis with fewer folds
python cross_validation.py --models DecisionTree KNN --n-splits 3

# Comprehensive analysis with all strategies
python cross_validation.py --cv-strategies KFold StratifiedKFold ShuffleSplit LeaveOneOut

# Custom random state for reproducibility
python cross_validation.py --random-state 123
```

### Structured Output System:
The `compare_models_structured.py` script organizes all outputs in a structured directory system:
```
outputs/
├── logs/                     # Training and detailed logs
├── results/
│   ├── model_comparison/     # Comparison summaries
│   ├── individual_models/    # Per-model results
│   ├── hyperparameter_tuning/ # Hyperparameter tuning results
│   ├── cross_validation/     # Cross-validation analysis results
│   └── testing/             # Model testing results
├── visualizations/
│   ├── confusion_matrices/   # Individual model confusion matrices
│   ├── comparison_plots/     # Model comparison charts
│   └── data_exploration/     # Data analysis plots
├── reports/
│   ├── classification/       # Classification reports
│   └── summary/             # Executive summaries
└── models/                  # Saved model artifacts
```

### Clean up old results:
```bash
# Clean files older than 7 days (dry run first)
python cleanup_results.py --dry-run --days 7

# Clean files older than 7 days
python cleanup_results.py --days 7

# Deep clean including Python cache files
python cleanup_results.py --days 7 --deep-clean

# Keep reports but clean other files
python cleanup_results.py --days 7 --keep-reports

# Clean MLflow artifacts older than 30 days
python cleanup_results.py --mlflow-days 30
```

## What you'll learn

- How to track experiments with MLflow using the Iris dataset
- How to log parameters, metrics, and models
- How to use MLflow projects for reproducible runs
- How to compare different experiments in the UI
- Classification of iris flowers into 3 species (setosa, versicolor, virginica)