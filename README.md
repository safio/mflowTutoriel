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

## What you'll learn

- How to track experiments with MLflow using the Iris dataset
- How to log parameters, metrics, and models
- How to use MLflow projects for reproducible runs
- How to compare different experiments in the UI
- Classification of iris flowers into 3 species (setosa, versicolor, virginica)