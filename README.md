# MLflow Tutorial: Iris Classification

A comprehensive machine learning tutorial project demonstrating MLflow experiment tracking with the Iris flower classification dataset. This project showcases modern ML workflows including model training, comparison, hyperparameter tuning, cross-validation, and structured output management.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [Model Comparison](#model-comparison)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Cross-Validation](#cross-validation)
  - [Model Testing](#model-testing)
- [MLflow Integration](#mlflow-integration)
- [Output Management](#output-management)
- [Common Commands](#common-commands)
- [Cleanup](#cleanup)
- [Troubleshooting](#troubleshooting)

## Features

- **8 Pre-configured ML Algorithms**: Random Forest, SVM (Linear/RBF), Logistic Regression, KNN, Decision Tree, Gradient Boosting, AdaBoost
- **Comprehensive Experiment Tracking**: Full MLflow integration with automatic metric and artifact logging
- **Hyperparameter Optimization**: Grid search and random search with MLflow tracking
- **Cross-Validation Analysis**: K-fold and stratified K-fold validation strategies
- **Structured Output Management**: Automated organization of all experiment outputs with timestamps
- **Model Comparison Tools**: Side-by-side performance analysis with visualizations
- **Modular Architecture**: Clean separation of concerns with reusable components

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)
- Git (for version control)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mflowTutoriel
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- MLflow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Quick Start

### 1. Start MLflow UI

```bash
python -m mlflow ui --host 0.0.0.0 --port 5001
```

Then open your browser to `http://localhost:5001` to view the MLflow dashboard.

### 2. Run Your First Training

```bash
python train.py --n_estimators 100 --max_depth 10
```

### 3. View Results

Check the MLflow UI to see:
- Experiment metrics (accuracy, precision, recall, F1-score)
- Model artifacts (saved models, confusion matrices)
- Training parameters and metadata

## Project Structure

```
mflowTutoriel/
├── src/
│   └── mlflow_tutorial/
│       ├── models/              # Model configurations and training utilities
│       │   ├── model_configs.py # Centralized model configs and hyperparameter grids
│       │   ├── model_registry.py # Model registration utilities
│       │   └── training.py      # Common training workflows
│       ├── data/                # Data preprocessing and management
│       │   ├── data_manager.py
│       │   └── preprocessing.py
│       └── experiments/         # Experiment tracking utilities
├── outputs/                     # All experiment outputs (auto-generated)
│   ├── logs/                   # Training and debug logs
│   ├── results/                # Experiment results
│   ├── visualizations/         # Plots and charts
│   ├── reports/                # Analysis reports
│   └── models/                 # Saved model artifacts
├── mlruns/                     # MLflow tracking data
├── train.py                    # Single model training script
├── compare_models_structured.py # Comprehensive model comparison
├── hyperparameter_tuning.py    # HP optimization with MLflow
├── cross_validation.py         # K-fold CV analysis
├── test_model.py               # Model testing utilities
├── cleanup_results.py          # Output cleanup tool
└── requirements.txt            # Python dependencies
```

## Usage

### Model Training

#### Train a Single Model

```bash
# Train with default parameters
python train.py

# Train with custom parameters
python train.py --n_estimators 200 --max_depth 15

# Train using MLflow project
python -m mlflow run . -P n_estimators=200 -P max_depth=15 --env-manager local --experiment-name iris-classification
```

### Model Comparison

Compare multiple models with structured output:

```bash
# Compare all available models
python compare_models_structured.py

# Compare specific models only
python compare_models.py --models SVM_Linear LogisticRegression KNN

# Use the advanced structured version with detailed logging
python compare_models_structured.py --models SVM_Linear LogisticRegression KNN

# Customize data split, random state, and output directory
python compare_models.py --test-size 0.3 --random-state 123 --output-dir my_outputs
```

This generates:
- Performance metrics comparison table
- Confusion matrices for each model
- Comparative visualizations
- Detailed analysis report

### Hyperparameter Tuning

Optimize model hyperparameters using grid or random search:

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

Results include:
- Best parameters for each model
- Performance across parameter space
- MLflow tracking of all trials

### Cross-Validation

Perform K-fold cross-validation analysis:

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

Outputs:
- Cross-validation scores for each fold
- Mean and standard deviation metrics
- Visualization of score distributions

### Model Testing

Test trained models on new data:

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

## MLflow Integration

### Experiment Tracking

All training runs are automatically tracked in MLflow with:

- **Parameters**: Model hyperparameters, data splits, random seeds
- **Metrics**: Accuracy, precision, recall, F1-score, training time
- **Artifacts**:
  - Trained model files (.pkl)
  - Confusion matrices (PNG)
  - Classification reports (TXT)
  - Feature importance plots

### Viewing Experiments

1. Start MLflow UI: `python -m mlflow ui --host 0.0.0.0 --port 5001`
2. Navigate to `http://localhost:5001`
3. Browse experiments, compare runs, and analyze results

### Experiment Names

- `iris-classification` - Single model training runs
- `iris-model-comparison` - Model comparison experiments
- `iris-hyperparameter-tuning` - HP optimization runs
- `iris-cross-validation` - Cross-validation experiments

## Output Management

All outputs are organized in the `outputs/` directory:

```
outputs/
├── logs/                          # Timestamped training logs
├── results/
│   ├── model_comparison/          # Comparison summaries and reports
│   ├── individual_models/         # Per-model detailed results
│   ├── hyperparameter_tuning/     # HP search results
│   ├── cross_validation/          # CV analysis outputs
│   └── testing/                   # Model testing results
├── visualizations/
│   ├── confusion_matrices/        # Individual model confusion matrices
│   ├── comparison_plots/          # Cross-model comparison charts
│   └── data_exploration/          # Dataset analysis plots
├── reports/
│   ├── classification/            # Classification reports
│   └── summary/                   # Executive summaries
└── models/                        # Saved model artifacts
```

Each output file includes timestamps for easy tracking and version management.

## Common Commands

### MLflow UI

```bash
# Standard startup (recommended)
python -m mlflow ui --host 0.0.0.0 --port 5001

# Using startup script
python start_mlflow.py
```

### Training

```bash
# Quick training
python train.py

# With parameters
python train.py --n_estimators 100 --max_depth 10

# MLflow project run
python -m mlflow run . -P n_estimators=200 --env-manager local
```

### Analysis

```bash
# Model comparison
python compare_models_structured.py

# Hyperparameter tuning
python hyperparameter_tuning.py --models RandomForest SVM_Linear

# Cross-validation
python cross_validation.py --models RandomForest SVM_Linear

# Model testing
python test_model.py --use-sample
```

## Cleanup

Remove old results and temporary files:

```bash
# Dry run (preview what will be deleted)
python cleanup_results.py --dry-run --days 7

# Delete outputs older than 7 days
python cleanup_results.py --days 7

# Deep clean including Python cache files
python cleanup_results.py --days 7 --deep-clean

# Keep reports but clean other files
python cleanup_results.py --days 7 --keep-reports

# Clean MLflow artifacts older than 30 days
python cleanup_results.py --mlflow-days 30
```

## Troubleshooting

### Common Issues

If you encounter errors like:
- `MlflowException: Could not create run under non-active experiment`
- MLflow UI showing deleted experiments
- Import errors or missing dependencies

**See the comprehensive [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide for detailed solutions.**

Quick fix for deleted experiment errors:
```bash
# Stop MLflow UI first, then:
rm -rf mlruns/.trash

# Or use the fix script:
python fix_mlflow_experiments.py
```

## What You'll Learn

- How to track experiments with MLflow using the Iris dataset
- How to log parameters, metrics, and models
- How to use MLflow projects for reproducible runs
- How to compare different experiments in the UI
- Classification of iris flowers into 3 species (setosa, versicolor, virginica)
- Best practices for ML experiment organization and management

## Data Requirements

This project uses the built-in Iris dataset from scikit-learn, which includes:
- 150 samples
- 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: setosa, versicolor, virginica

For custom data testing, provide CSV files with the same feature structure.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Add your license information here]

## Acknowledgments

- MLflow for experiment tracking capabilities
- scikit-learn for machine learning algorithms
- The Iris dataset from R.A. Fisher's 1936 paper

## Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Happy Machine Learning!** 🚀
