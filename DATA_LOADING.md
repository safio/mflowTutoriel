# Data Loading Guide

This document explains how data is loaded and prepared for training in the MLflow Tutorial project.

## Table of Contents

- [Overview](#overview)
- [Data Loading Approaches](#data-loading-approaches)
  - [Direct Loading (Most Common)](#1-direct-loading-approach-most-common)
  - [DataManager Class (Modular)](#2-datamanager-class-approach-modular)
- [Data Loading Workflow](#data-loading-workflow)
- [Dataset Details](#dataset-details)
- [Loading Scenarios](#different-loading-scenarios)
- [Preprocessing Options](#preprocessing-options)
- [Configuration Parameters](#configuration-parameters)

---

## Overview

The project uses **two approaches** for loading data:

1. **Direct Loading** - Simple, inline data loading (used in most scripts)
2. **DataManager Class** - Modular, reusable data management (available in `src/mlflow_tutorial/data/`)

All training uses the **Iris flower dataset** from scikit-learn, which is included with the library - no external data files are required.

---

## Data Loading Approaches

### 1. Direct Loading Approach (Most Common)

**Used in:** `train.py`, `compare_models_structured.py`, `hyperparameter_tuning.py`, `cross_validation.py`

This approach loads data directly using sklearn and pandas in each script.

#### Pattern

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data(test_size=0.2, random_state=42):
    """Load and prepare the iris dataset"""
    # Step 1: Load iris dataset from sklearn
    iris = load_iris()

    # Step 2: Convert to pandas DataFrame for easier handling
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Step 3: Split into train/test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Ensures balanced class distribution
    )

    return X_train, X_test, y_train, y_test, iris.target_names
```

#### Example from `train.py`

Location: `train.py:23-36`

```python
# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print(f"Dataset info:")
print(f"- Total samples: {len(X)}")
print(f"- Features: {list(X.columns)}")
print(f"- Target classes: {iris.target_names}")
print(f"- Class distribution: {pd.Series(y).value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=args.random_state
)
```

#### Why This Approach?

- ✅ Simple and straightforward
- ✅ Easy to understand for beginners
- ✅ Self-contained in each script
- ✅ No additional imports needed
- ✅ Direct control over parameters

---

### 2. DataManager Class Approach (Modular)

**Location:** `src/mlflow_tutorial/data/data_manager.py`

This approach provides an object-oriented interface for data management with built-in logging and reusable methods.

#### Class Overview

```python
from src.mlflow_tutorial.data.data_manager import DataManager

# Initialize the manager (automatically loads iris data)
data_manager = DataManager()
```

#### Key Methods

##### `get_data()`
Returns the full dataset as a tuple of (features, targets).

```python
X, y = data_manager.get_data()
# X: pandas DataFrame with 150 rows, 4 columns
# y: numpy array with 150 class labels (0, 1, 2)
```

##### `get_train_test_split()`
Returns train/test split with stratification.

```python
X_train, X_test, y_train, y_test = data_manager.get_train_test_split(
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    stratify=True       # Balanced class distribution (default)
)
```

##### `get_feature_info()`
Returns comprehensive dataset information.

```python
info = data_manager.get_feature_info()
```

Returns a dictionary with:
```python
{
    'feature_names': ['sepal length (cm)', 'sepal width (cm)', ...],
    'target_names': ['setosa', 'versicolor', 'virginica'],
    'n_samples': 150,
    'n_features': 4,
    'n_classes': 3,
    'feature_stats': {
        'sepal length (cm)': {'mean': 5.84, 'std': 0.83, ...},
        ...
    },
    'class_distribution': {0: 50, 1: 50, 2: 50}
}
```

##### `get_sample_data()`
Returns random samples for testing.

```python
sample = data_manager.get_sample_data(n_samples=5)
# Returns: pandas DataFrame with 5 random samples
```

##### `save_sample_data()`
Exports sample data to CSV for external testing.

```python
data_manager.save_sample_data("sample_test_data.csv", n_samples=10)
# Creates: CSV file with 10 random samples
```

#### Full Example

```python
from src.mlflow_tutorial.data.data_manager import DataManager

# Initialize
dm = DataManager()

# Get dataset information
info = dm.get_feature_info()
print(f"Dataset: {info['n_samples']} samples, {info['n_features']} features")
print(f"Classes: {info['target_names']}")

# Get train/test split
X_train, X_test, y_train, y_test = dm.get_train_test_split(
    test_size=0.3,
    random_state=123
)

# Save sample for testing
dm.save_sample_data("my_test_data.csv", n_samples=20)
```

#### Advantages

- ✅ Centralized data management
- ✅ Built-in logging
- ✅ Reusable across multiple scripts
- ✅ Consistent interface
- ✅ Easy to extend with new methods

---

## Data Loading Workflow

```
┌─────────────────────────────────────────────────┐
│  1. Load Iris Dataset from sklearn             │
│     - 150 samples (50 per class)                │
│     - 4 features (sepal/petal length/width)     │
│     - 3 classes (setosa, versicolor, virginica) │
│     Source: sklearn.datasets.load_iris()        │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  2. Convert to Pandas DataFrame                 │
│     Features: DataFrame with column names       │
│     - sepal length (cm)                         │
│     - sepal width (cm)                          │
│     - petal length (cm)                         │
│     - petal width (cm)                          │
│     Target: numpy array [0, 1, 2]               │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  3. Split Data (train_test_split)              │
│     - Default: 80% train / 20% test            │
│     - Stratified: maintains class balance       │
│     - Random state: 42 (reproducibility)        │
│     Results:                                    │
│     - X_train: 120 samples                      │
│     - X_test:  30 samples                       │
│     - y_train: 120 labels                       │
│     - y_test:  30 labels                        │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  4. Return Split Data                           │
│     X_train (120 samples) → Model Training      │
│     X_test  (30 samples)  → Model Evaluation    │
│     y_train, y_test       → Target labels       │
│     target_names          → ['setosa', ...]     │
└─────────────────────────────────────────────────┘
```

---

## Dataset Details

### Iris Dataset Characteristics

| Property | Value |
|----------|-------|
| **Total Samples** | 150 |
| **Features** | 4 (continuous, measured in cm) |
| **Classes** | 3 (iris species) |
| **Class Distribution** | Perfectly balanced (50 samples each) |
| **Missing Values** | None |
| **Feature Units** | Centimeters (cm) |
| **Target Format** | Integer (0, 1, 2) |
| **Target Names** | setosa, versicolor, virginica |

### Feature Descriptions

1. **sepal length (cm)** - Length of the sepal (outer part of the flower)
2. **sepal width (cm)** - Width of the sepal
3. **petal length (cm)** - Length of the petal (inner part of the flower)
4. **petal width (cm)** - Width of the petal

### Target Classes

| Class ID | Class Name | Samples | Characteristics |
|----------|-----------|---------|-----------------|
| 0 | Setosa | 50 | Smallest petals, easiest to classify |
| 1 | Versicolor | 50 | Medium-sized, some overlap with virginica |
| 2 | Virginica | 50 | Largest petals, longest sepals |

### Example Data Structure

```python
# Features (X) - pandas DataFrame
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2

# Target (y) - numpy array
array([0, 0, 0, 1, 1, 1, 2, 2, 2, ...])

# Target names
array(['setosa', 'versicolor', 'virginica'])
```

### Feature Statistics

| Feature | Min | Max | Mean | Std Dev |
|---------|-----|-----|------|---------|
| sepal length (cm) | 4.3 | 7.9 | 5.84 | 0.83 |
| sepal width (cm) | 2.0 | 4.4 | 3.05 | 0.43 |
| petal length (cm) | 1.0 | 6.9 | 3.76 | 1.76 |
| petal width (cm) | 0.1 | 2.5 | 1.20 | 0.76 |

---

## Different Loading Scenarios

### Scenario A: Training Scripts

**Scripts:** `train.py`, `compare_models_structured.py`, `hyperparameter_tuning.py`, `cross_validation.py`

**Process:**
1. Load built-in Iris dataset from sklearn
2. Split into train/test sets
3. Train models on training set
4. Evaluate on test set

**No external files needed** - everything is built-in.

```python
# Standard training data loading
def prepare_data(test_size=0.2, random_state=42):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, iris.target_names
```

### Scenario B: Model Testing

**Script:** `test_model.py`

Three options for loading test data:

#### Option 1: Built-in Sample Data (Hardcoded)

```python
def create_sample_data():
    """Create sample data points for testing"""
    sample_data = pd.DataFrame({
        'sepal length (cm)': [5.1, 6.2, 4.9, 7.0, 5.8],
        'sepal width (cm)': [3.5, 2.9, 3.0, 3.2, 2.7],
        'petal length (cm)': [1.4, 4.3, 1.4, 4.7, 4.1],
        'petal width (cm)': [0.2, 1.3, 0.2, 1.4, 1.0]
    })
    return sample_data

# Usage
test_data = create_sample_data()
```

**Command:**
```bash
python test_model.py --use-sample
```

#### Option 2: CSV File (Custom Data)

```python
# Load from external CSV file
test_data = pd.read_csv(args.data_file)
```

**Command:**
```bash
python test_model.py --data-file sample_test_data.csv
```

**CSV Format:**
```csv
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
5.1,3.5,1.4,0.2
6.2,2.9,4.3,1.3
```

#### Option 3: Original Dataset Samples

```python
# Use first N samples from original iris dataset
iris = load_iris()
test_data = pd.DataFrame(iris.data[:5], columns=iris.feature_names)
```

**Command:**
```bash
python test_model.py  # Default behavior
```

### Scenario C: Data Exploration

**Script:** `explore_data.py`

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load full dataset for analysis
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Combine for exploration
data = X.copy()
data['species'] = [iris.target_names[target] for target in y]
data['class_number'] = y

# Analyze patterns, visualize distributions, etc.
```

**Command:**
```bash
python explore_data.py
```

---

## Preprocessing Options

### Available Preprocessing Tools

**Location:** `src/mlflow_tutorial/data/preprocessing.py`

The `DataPreprocessor` class provides advanced preprocessing capabilities:

#### 1. Feature Scaling

```python
from src.mlflow_tutorial.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

# Standard scaling (zero mean, unit variance)
X_train_scaled, X_test_scaled = preprocessor.scale_features(
    X_train, X_test, method='standard'
)

# Min-Max scaling (scale to [0, 1])
X_train_scaled, X_test_scaled = preprocessor.scale_features(
    X_train, X_test, method='minmax'
)
```

#### 2. Feature Selection

```python
# Select top k features using statistical tests
X_train_selected, X_test_selected = preprocessor.select_features(
    X_train, y_train, X_test, k=3
)

# Get feature importance scores
scores = preprocessor.get_feature_importance_scores()
# Returns: {'sepal length (cm)': 119.26, 'petal width (cm)': 960.01, ...}
```

#### 3. Label Encoding

```python
# Convert string targets to integers (if needed)
y_encoded = preprocessor.encode_targets(y)
```

#### 4. Polynomial Features

```python
# Create interaction terms
X_poly = preprocessor.create_polynomial_features(
    X, degree=2, include_bias=False
)
# Increases features from 4 to 14 (with degree=2)
```

#### 5. Missing Value Handling

```python
# Impute missing values
X_imputed = preprocessor.handle_missing_values(
    X, strategy='mean'  # or 'median', 'mode'
)
```

### Note on Preprocessing

**Important:** These preprocessing tools are **currently not used** in the main training scripts because:
- The Iris dataset has no missing values
- Features are already well-scaled
- All 4 features are informative

These tools are provided for:
- Advanced experimentation
- Custom datasets
- Learning preprocessing techniques

---

## Configuration Parameters

### Key Parameters

| Parameter | Default | Purpose | Configurable In |
|-----------|---------|---------|-----------------|
| `test_size` | 0.2 | Proportion of data for testing (20%) | All training scripts |
| `random_state` | 42 | Random seed for reproducibility | All scripts |
| `stratify` | `y` (True) | Ensure balanced class distribution in splits | Most scripts |
| `n_samples` | 5-10 | Number of samples for testing/demo | DataManager, test scripts |

### Modifying Parameters

#### Command Line Arguments

Most scripts accept command-line arguments:

```bash
# train.py
python train.py --random_state 123

# compare_models_structured.py
python compare_models_structured.py --test-size 0.3 --random-state 999

# hyperparameter_tuning.py
python hyperparameter_tuning.py --test-size 0.25
```

#### In Code

```python
# Modify in prepare_data() function
X_train, X_test, y_train, y_test = prepare_data(
    test_size=0.3,      # 30% test set
    random_state=123    # Different seed
)

# Or with DataManager
dm = DataManager()
X_train, X_test, y_train, y_test = dm.get_train_test_split(
    test_size=0.25,
    random_state=99,
    stratify=True
)
```

### Recommended Settings

| Use Case | test_size | random_state | stratify |
|----------|-----------|--------------|----------|
| **Quick Testing** | 0.2 | 42 | True |
| **Production** | 0.2-0.3 | 42 | True |
| **Cross-Validation** | N/A (CV handles splits) | 42 | True |
| **Small Datasets** | 0.2 | 42 | True |
| **Large Datasets** | 0.1-0.15 | 42 | True |

### Why Stratify?

Stratification ensures each split has the same class distribution:

**Without Stratification:**
- Train: 70% setosa, 20% versicolor, 10% virginica
- Test: 10% setosa, 30% versicolor, 60% virginica

**With Stratification:**
- Train: 33% setosa, 33% versicolor, 33% virginica
- Test: 33% setosa, 33% versicolor, 33% virginica

This prevents biased model evaluation.

---

## Summary

### Current Implementation

✅ **Simple and Straightforward**
- Direct loading from sklearn's built-in dataset
- No external files or dependencies required
- Consistent across all training scripts

✅ **Well-Structured**
- Clear data flow: load → convert → split → train
- Stratified splitting ensures balanced class distribution
- Reproducible with fixed random seeds

✅ **Flexible**
- Modular DataManager class available for advanced use
- Preprocessing tools ready for custom datasets
- Multiple testing scenarios supported

### Data Flow Summary

```
sklearn.datasets.load_iris()
    ↓
pandas.DataFrame (150 samples, 4 features)
    ↓
train_test_split() with stratification
    ↓
X_train (120), X_test (30), y_train (120), y_test (30)
    ↓
Model Training & Evaluation
```

### Quick Reference Commands

```bash
# View dataset information
python explore_data.py

# Train with default settings
python train.py

# Train with custom split
python train.py --random_state 123

# Test model with sample data
python test_model.py --use-sample

# Test model with CSV file
python test_model.py --data-file my_data.csv

# Compare models with custom split
python compare_models_structured.py --test-size 0.3
```

---

## Additional Resources

### Related Files

- `src/mlflow_tutorial/data/data_manager.py` - DataManager class implementation
- `src/mlflow_tutorial/data/preprocessing.py` - Preprocessing utilities
- `explore_data.py` - Dataset exploration and visualization
- `test_model.py` - Model testing with various data sources

### External Documentation

- [Iris Dataset Documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
- [sklearn.datasets.load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

---

**Last Updated:** 2025-10-24
**Maintainer:** Project Team
