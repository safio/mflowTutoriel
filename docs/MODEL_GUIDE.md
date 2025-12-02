# Machine Learning Models Guide

**A practical guide to understanding and using ML models in real-world projects**

---

## Table of Contents
1. [Understanding the Models](#understanding-the-models)
2. [Understanding F1 Score](#understanding-f1-score)
3. [Quick Decision Guide](#quick-decision-guide)
4. [Practical Examples](#practical-examples)
5. [Running Comparisons](#running-comparisons)

---

## Understanding the Models

### 1. RandomForest
**Location in code:** `compare_models.py:36-38` | `model_configs.py:24-28`

**What it is:**
Combines multiple decision trees and averages their predictions (ensemble learning). Think of it as asking multiple experts and taking a vote.

**When to use in real life:**
- ✅ Medical diagnosis (predicting diseases)
- ✅ Customer churn prediction
- ✅ Credit risk assessment
- ✅ Default choice for many classification problems
- ✅ When you need good accuracy without much tuning

**Pros:**
- Handles missing data well
- Reduces overfitting
- Works with both numerical and categorical data
- Robust to outliers
- Can rank feature importance

**Cons:**
- Slower prediction time compared to simpler models
- Harder to interpret than single decision trees
- Large memory footprint

**Key Parameters:**
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum depth of trees (default: None)

**Example Use Cases:**
- **Banking:** Credit default prediction with 95% accuracy
- **Healthcare:** Patient readmission prediction
- **E-commerce:** Product recommendation engines

---

### 2. SVM_RBF (Support Vector Machine - RBF kernel)
**Location in code:** `compare_models.py:40-42` | `model_configs.py:29-33`

**What it is:**
Finds the best boundary between classes using complex non-linear patterns. RBF (Radial Basis Function) kernel allows it to create circular/curved decision boundaries.

**When to use in real life:**
- ✅ Image classification (facial recognition)
- ✅ Text categorization
- ✅ Handwriting recognition
- ✅ When you have complex, non-linear relationships
- ✅ Small to medium datasets (< 10,000 samples)

**Pros:**
- Excellent for complex patterns
- Memory efficient
- Works well in high-dimensional spaces
- Effective when number of dimensions > number of samples

**Cons:**
- Slow on large datasets (> 50,000 samples)
- Requires feature scaling (normalization)
- Difficult to interpret
- Sensitive to parameter tuning (C and gamma)

**Key Parameters:**
- `C`: Regularization parameter (default: 1.0)
- `gamma`: Kernel coefficient (default: 'scale')
- `kernel`: 'rbf' for non-linear boundaries

**Example Use Cases:**
- **Security:** Biometric authentication systems
- **Medicine:** Cancer cell classification from images
- **Finance:** Stock market prediction with complex patterns

---

### 3. SVM_Linear
**Location in code:** `compare_models.py:44-46` | `model_configs.py:34-38`

**What it is:**
Finds the best straight line (or hyperplane) to separate classes. Think of drawing a line between two groups.

**When to use in real life:**
- ✅ Spam email detection
- ✅ Sentiment analysis (positive/negative reviews)
- ✅ Document classification
- ✅ When features are already linearly separable
- ✅ Text-based problems with many features

**Pros:**
- Fast training and prediction
- Works well with high-dimensional data
- Memory efficient
- Good for text classification (with TF-IDF features)

**Cons:**
- Not good for complex non-linear patterns
- Requires feature scaling
- Sensitive to outliers

**Key Parameters:**
- `C`: Regularization parameter (default: 1.0)
- `kernel`: 'linear' for straight boundaries

**Example Use Cases:**
- **Email:** Gmail spam filter (processes millions of emails)
- **Social Media:** Sentiment analysis on tweets
- **News:** Automatic article categorization

---

### 4. LogisticRegression
**Location in code:** `compare_models.py:48-50` | `model_configs.py:39-43`

**What it is:**
Despite the name, it's a classification algorithm! Predicts probability of belonging to a class using a logistic function.

**When to use in real life:**
- ✅ Marketing: Will customer buy? (Yes/No)
- ✅ Healthcare: Disease present or not?
- ✅ Finance: Loan approval/rejection
- ✅ **When you need probability scores** (not just predictions)
- ✅ **When you need to explain predictions to stakeholders**

**Pros:**
- Very fast training and prediction
- Highly interpretable (you can see feature weights)
- Provides probability scores (0.0 to 1.0)
- Works well with small datasets
- No hyperparameter tuning required for basic use

**Cons:**
- Assumes linear relationships
- Not suitable for complex patterns
- Can underfit on complex datasets

**Key Parameters:**
- `C`: Inverse regularization strength (default: 1.0)
- `max_iter`: Maximum iterations (default: 1000)

**Example Use Cases:**
- **Insurance:** Claim approval (need to explain why to customers)
- **HR:** Resume screening (explain hiring decisions)
- **Medicine:** Disease risk assessment with probability

---

### 5. DecisionTree
**Location in code:** `compare_models.py:52-54` | `model_configs.py:44-48`

**What it is:**
Makes decisions using a series of yes/no questions (like a flowchart). Creates a tree structure where each node asks a question about a feature.

**When to use in real life:**
- ✅ Customer segmentation
- ✅ Risk assessment
- ✅ **When you need to explain decisions to non-technical people**
- ✅ Rule-based decision making
- ✅ When features have clear thresholds

**Pros:**
- Easy to visualize and explain
- No feature scaling required
- Handles both numerical and categorical data
- Handles non-linear relationships
- Can be exported as rules

**Cons:**
- Can overfit easily (memorizes training data)
- Unstable (small changes in data affect results)
- Biased toward features with more levels
- Generally lower accuracy than ensemble methods

**Key Parameters:**
- `max_depth`: Maximum depth of tree (default: None)
- `min_samples_split`: Minimum samples to split node (default: 2)

**Example Use Cases:**
- **Business:** Customer segmentation (VIP vs Regular)
- **Legal:** Risk assessment with clear rules
- **Education:** Student performance prediction

---

### 6. KNN (K-Nearest Neighbors)
**Location in code:** `compare_models.py:56-58` | `model_configs.py:49-53`

**What it is:**
Classifies based on what the closest K neighbors are. "You are the average of your 5 closest friends."

**When to use in real life:**
- ✅ Recommendation systems (similar products/users)
- ✅ Pattern recognition
- ✅ Anomaly detection
- ✅ Small datasets
- ✅ When new data is similar to training data

**Pros:**
- Simple and intuitive
- No training time (lazy learner)
- Adapts as new data is added
- Works well for multi-class problems

**Cons:**
- Slow predictions on large datasets
- Sensitive to irrelevant features
- Requires feature scaling
- High memory usage (stores all training data)
- Curse of dimensionality (struggles with many features)

**Key Parameters:**
- `n_neighbors`: Number of neighbors to consider (default: 5)
- `weights`: 'uniform' or 'distance' weighted

**Example Use Cases:**
- **E-commerce:** "Customers who bought this also bought..."
- **Music:** Song recommendation (Spotify-style)
- **Security:** Anomaly detection in network traffic

---

### 7. NaiveBayes
**Location in code:** `compare_models.py:60-62` | `model_configs.py:54-58`

**What it is:**
Uses probability theory assuming features are independent (naive assumption). Based on Bayes' theorem.

**When to use in real life:**
- ✅ **Spam filtering** (most common and effective use)
- ✅ Text classification
- ✅ Sentiment analysis
- ✅ Real-time prediction (very fast)
- ✅ When you have limited training data

**Pros:**
- Extremely fast (training and prediction)
- Works well with small datasets
- Excellent for text classification
- Handles high-dimensional data well
- Not sensitive to irrelevant features

**Cons:**
- Assumes feature independence (often not true in reality)
- Can be outperformed by more sophisticated models
- Zero-frequency problem (if a category + feature combo wasn't seen in training)

**Key Parameters:**
- `var_smoothing`: Portion of largest variance added to variances (default: 1e-9)

**Example Use Cases:**
- **Email:** Spam filters (Gmail, Outlook)
- **News:** Real-time article categorization
- **Social Media:** Content moderation (flag inappropriate content)

---

### 8. GradientBoosting
**Location in code:** `compare_models.py:64-66` | `model_configs.py:59-66`

**What it is:**
Builds models sequentially, each correcting the previous one's errors. Like a team where each member fixes mistakes from the previous person.

**When to use in real life:**
- ✅ Kaggle competitions (often wins!)
- ✅ Click-through rate prediction
- ✅ Search ranking (Google, Bing)
- ✅ When you need maximum accuracy
- ✅ Complex business problems with tabular data

**Pros:**
- Often the most accurate algorithm
- Handles complex patterns and interactions
- Robust to outliers
- Built-in feature importance
- Handles mixed data types

**Cons:**
- Slower to train
- Can overfit if not tuned properly
- Harder to interpret than simple models
- Requires careful hyperparameter tuning
- Sensitive to noisy data

**Key Parameters:**
- `n_estimators`: Number of boosting stages (default: 100)
- `learning_rate`: Shrinks contribution of each tree (default: 0.1)
- `max_depth`: Maximum depth of trees (default: 3)

**Example Use Cases:**
- **Search Engines:** Ranking search results (LambdaMART)
- **Advertising:** Click prediction for ad placement
- **Finance:** Credit scoring with maximum accuracy

---

## Understanding F1 Score

### What is F1 Score?

**F1 Score** is the harmonic mean of **Precision** and **Recall**. It provides a single score that balances both metrics.

**Formula:**
```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Location in code:** `compare_models.py:77` (f1_macro), `compare_models.py:80` (f1_weighted)

---

### Simple Analogy: Fishing 🎣

- **Precision:** Of all the fish you caught, how many were the right species?
- **Recall:** Of all the right fish in the lake, how many did you catch?
- **F1 Score:** The balance between precision and recall

---

### Real-World Example: Medical Testing

Let's say you're building a model to detect cancer:

```
Scenario:
- 100 patients tested
- 10 actually have cancer (ground truth)
- Your model predicts 15 have cancer
- 8 of those 15 predictions are correct
```

**Calculations:**
- **Precision = 8/15 = 53.3%**
  (Of patients you flagged, 53.3% actually have cancer)

- **Recall = 8/10 = 80.0%**
  (Of patients with cancer, you caught 80%)

- **F1 Score = 2 × (0.533 × 0.80) / (0.533 + 0.80) = 0.64 or 64%**

---

### Why F1 Score Matters

#### Use F1 when classes are imbalanced:

1. **Fraud Detection** (99% legitimate, 1% fraud)
   - Accuracy alone is misleading!
   - Model predicting "all legitimate" = 99% accuracy ❌ (catches no fraud!)
   - Good model with 85% F1 = catches actual fraud ✅

2. **Disease Detection** (rare diseases)
   - Missing sick patients is critical (high recall needed)
   - False alarms are costly (high precision needed)
   - F1 balances both concerns

3. **Spam Detection** (mostly real emails)
   - Don't want to miss important emails (recall)
   - Don't want false spam flags (precision)

---

### F1 Variations in Your Code

**1. F1 Macro** (compare_models.py:77)
- Calculates F1 for each class separately, then averages
- Treats all classes equally (good for imbalanced data)
- **Use when:** All classes are equally important

**2. F1 Weighted** (compare_models.py:80)
- Calculates F1 for each class, weighted by class frequency
- Accounts for class imbalance
- **Use when:** Some classes are more common than others

**3. F1 Per-Class** (compare_models.py:92)
- Shows F1 for each individual class (setosa, versicolor, virginica)
- **Use when:** You want to see performance on specific classes

---

### When to Prioritize Different Metrics

| Scenario | Metric to Prioritize | Why |
|----------|---------------------|-----|
| Fraud detection | **Recall** (or F1) | Missing fraud is very costly |
| Spam filtering | **Precision** (or F1) | False positives annoy users |
| Disease screening | **Recall** (or F1) | Missing sick patients is critical |
| Ad click prediction | **F1** or **AUC** | Balance between showing ads and relevance |
| Balanced dataset | **Accuracy** | All classes equally important |
| Imbalanced dataset | **F1 Score** | Accuracy can be misleading |

---

## Quick Decision Guide

### Start Here: Which Model Should I Use?

```
START HERE
│
├─ Need to explain predictions?
│  └─ LogisticRegression, DecisionTree
│
├─ Text classification?
│  └─ NaiveBayes, SVM_Linear
│
├─ Image recognition?
│  └─ SVM_RBF, RandomForest
│
├─ Need maximum accuracy?
│  └─ GradientBoosting, RandomForest
│
├─ Have small dataset (< 1000 samples)?
│  └─ SVM_RBF, KNN, NaiveBayes
│
├─ Need fast predictions?
│  └─ LogisticRegression, NaiveBayes
│
├─ Have large dataset (> 100,000 samples)?
│  └─ LogisticRegression, RandomForest
│
└─ Not sure?
   └─ **Start with RandomForest** (best general-purpose model)
```

---

### Model Recommendations by Task Type

**From `model_configs.py:143-159`**

#### High Accuracy Projects
```python
Recommended: RandomForest, GradientBoosting, SVM_RBF
```
- Kaggle competitions
- Research projects
- When accuracy is paramount

#### Interpretable/Explainable AI
```python
Recommended: DecisionTree, LogisticRegression, NaiveBayes
```
- Healthcare (need to explain diagnoses)
- Finance (regulatory requirements)
- Legal (justify decisions)

#### Fast Inference (Production Systems)
```python
Recommended: LogisticRegression, NaiveBayes, KNN
```
- Real-time applications
- High-traffic websites
- Mobile applications

#### Small Dataset (< 1000 samples)
```python
Recommended: SVM_RBF, KNN, NaiveBayes
```
- Medical research with limited data
- Rare event prediction
- Startup MVP with limited data

---

### Model Selection Matrix

| Criterion | Best Models | Avoid |
|-----------|-------------|-------|
| **Interpretability** | LogisticRegression, DecisionTree | GradientBoosting, SVM_RBF |
| **Speed** | NaiveBayes, LogisticRegression | GradientBoosting, SVM_RBF |
| **Accuracy** | GradientBoosting, RandomForest | DecisionTree, NaiveBayes |
| **Small Data** | SVM_RBF, NaiveBayes | GradientBoosting, RandomForest |
| **Large Data** | LogisticRegression, RandomForest | KNN, SVM_RBF |
| **High Dimensions** | SVM_Linear, LogisticRegression | KNN, DecisionTree |
| **Non-linear Patterns** | SVM_RBF, RandomForest | LogisticRegression, SVM_Linear |

---

## Practical Examples

### Example 1: Email Spam Detection

**Problem:** Classify emails as spam or not spam
**Dataset:** 50,000 emails, 10% spam (imbalanced)

**Best Model Choice:** `NaiveBayes` or `SVM_Linear`

**Why?**
- Text classification (Naive Bayes excels here)
- Need fast real-time prediction
- High-dimensional features (TF-IDF)
- Imbalanced data (use F1 score)

**Metrics to Track:**
- **Precision:** Don't flag real emails as spam (annoying!)
- **Recall:** Catch most spam
- **F1 Score:** Balance both

---

### Example 2: Credit Default Prediction

**Problem:** Will customer default on loan?
**Dataset:** 10,000 customers, 5% default rate

**Best Model Choice:** `RandomForest` or `GradientBoosting`

**Why?**
- Need high accuracy (money at stake)
- Complex patterns in financial data
- Need feature importance (which factors matter most)
- Imbalanced data

**Metrics to Track:**
- **Recall:** Don't miss likely defaulters (costly!)
- **F1 Score:** Balance precision and recall
- **AUC-ROC:** Evaluate threshold tuning

---

### Example 3: Medical Diagnosis

**Problem:** Detect diabetes from patient data
**Dataset:** 500 patients, 20% diabetic

**Best Model Choice:** `LogisticRegression` or `RandomForest`

**Why?**
- Need to explain to doctors (interpretability)
- Provide probability scores (risk assessment)
- Small dataset
- Regulatory requirements

**Metrics to Track:**
- **Recall:** Don't miss sick patients (critical!)
- **F1 Score:** Balance false positives/negatives
- **Sensitivity/Specificity:** Medical standard metrics

---

### Example 4: Image Classification

**Problem:** Classify flower species from images
**Dataset:** 10,000 images, 100 species

**Best Model Choice:** `SVM_RBF` or `RandomForest`

**Why?**
- Non-linear patterns in image features
- Multi-class classification (100 classes)
- Medium-sized dataset
- Complex feature interactions

**Metrics to Track:**
- **Accuracy:** Balanced classes
- **F1 Macro:** Performance across all species
- **Confusion Matrix:** See which species are confused

---

### Example 5: Customer Churn Prediction

**Problem:** Which customers will cancel subscription?
**Dataset:** 100,000 customers, 15% churn

**Best Model Choice:** `GradientBoosting` or `RandomForest`

**Why?**
- Need high accuracy (revenue impact)
- Large dataset available
- Complex customer behavior patterns
- Need feature importance (what drives churn)

**Metrics to Track:**
- **Recall:** Catch customers about to churn
- **Precision:** Don't waste resources on false positives
- **F1 Score:** Balance intervention costs

---

## Running Comparisons

### Basic Comparison (All Models)

```bash
python compare_models.py
```

**Output Location:** `outputs/results/model_comparison_results.csv`
**Visualizations:** `outputs/visualizations/comparison_plots/model_comparison.png`

---

### Compare Specific Models

```bash
# Compare just 3 models
python compare_models.py --models RandomForest SVM_RBF LogisticRegression

# Compare ensemble models only
python compare_models.py --models RandomForest GradientBoosting

# Compare fast models
python compare_models.py --models LogisticRegression NaiveBayes KNN
```

---

### Structured Comparison (Recommended)

```bash
python compare_models_structured.py
```

**Features:**
- Organized output directory structure
- Comprehensive logging
- Individual model reports
- Comparison visualizations
- MLflow experiment tracking

---

### Hyperparameter Tuning

```bash
# Tune specific models
python hyperparameter_tuning.py --models RandomForest SVM_Linear

# Random search (faster)
python hyperparameter_tuning.py --search-type random --n-iter 50

# Grid search (exhaustive)
python hyperparameter_tuning.py --search-type grid
```

---

### Cross-Validation Analysis

```bash
# 5-fold CV on multiple models
python cross_validation.py --models RandomForest SVM_Linear KNN

# Try different CV strategies
python cross_validation.py --cv-strategies KFold StratifiedKFold
```

---

### View Results in MLflow

```bash
# Start MLflow UI
python -m mlflow ui --host 0.0.0.0 --port 5001

# Open browser to: http://localhost:5001
```

**What to look for:**
1. **Accuracy:** Overall correctness
2. **F1 Macro:** Performance across all classes
3. **CV Accuracy:** Model consistency
4. **Training Time:** Efficiency
5. **Confusion Matrix:** Where model makes mistakes

---

## Understanding Your Results

### Reading the Comparison Output

When you run `compare_models.py`, you'll see output like:

```
🔄 Training RandomForest...
✅ RandomForest Results:
   Accuracy: 0.9667
   F1-Score (macro): 0.9662
   CV Accuracy: 0.9500 ± 0.0447
   Training Time: 0.234s
```

**What this means:**
- **Accuracy: 0.9667** → 96.67% of predictions are correct
- **F1-Score: 0.9662** → Balanced precision/recall across all classes
- **CV Accuracy: 0.95 ± 0.04** → Model is consistent (95% ± 4%)
- **Training Time: 0.234s** → Fast training

---

### Interpreting F1 Scores

| F1 Score | Interpretation | Action |
|----------|----------------|--------|
| 0.90 - 1.00 | Excellent | Model ready for production |
| 0.80 - 0.90 | Good | May need minor tuning |
| 0.70 - 0.80 | Fair | Needs improvement |
| 0.60 - 0.70 | Poor | Try different model or features |
| < 0.60 | Very Poor | Rethink approach |

---

### Common Patterns to Look For

**1. High Accuracy, Low F1:**
- Model is biased toward majority class
- Check class distribution
- Use class weights or resampling

**2. High Training Time, Low Accuracy:**
- Model too complex for data
- Try simpler model
- Check for data quality issues

**3. Large CV Standard Deviation:**
- Model is unstable
- Need more data or simpler model
- Check for data leakage

**4. Good CV, Poor Test:**
- Overfitting!
- Reduce model complexity
- Add regularization

---

## Best Practices

### 1. Always Start Simple
```python
# Start with LogisticRegression or RandomForest
# Then try more complex models if needed
```

### 2. Check Data Balance
```python
# If imbalanced, focus on F1 Score, not just Accuracy
```

### 3. Use Cross-Validation
```python
# Never trust a single train/test split
# Always use CV (k=5 or k=10)
```

### 4. Track Everything in MLflow
```python
# Log parameters, metrics, and artifacts
# Compare runs systematically
```

### 5. Visualize Results
```python
# Always check confusion matrices
# Look for patterns in errors
```

---

## Additional Resources

### In This Repository

- **Model Configurations:** `src/mlflow_tutorial/models/model_configs.py`
- **Model Comparison:** `compare_models.py`, `compare_models_structured.py`
- **Hyperparameter Tuning:** `hyperparameter_tuning.py`
- **Cross-Validation:** `cross_validation.py`
- **Troubleshooting Guide:** `docs/TROUBLESHOOTING.md`
- **Data Loading Guide:** `docs/DATA_LOADING.md`

### Recommended Reading

1. **Scikit-learn Documentation:**
   https://scikit-learn.org/stable/user_guide.html

2. **MLflow Documentation:**
   https://mlflow.org/docs/latest/index.html

3. **Model Selection:**
   https://scikit-learn.org/stable/tutorial/machine_learning_map/

4. **Metrics Guide:**
   https://scikit-learn.org/stable/modules/model_evaluation.html

---

## Quick Reference Commands

```bash
# Compare all models
python compare_models.py

# Compare specific models
python compare_models.py --models RandomForest SVM_RBF

# Structured comparison (recommended)
python compare_models_structured.py

# Hyperparameter tuning
python hyperparameter_tuning.py --models RandomForest

# Cross-validation
python cross_validation.py --models RandomForest SVM_Linear

# Start MLflow UI
python -m mlflow ui --port 5001

# Test a trained model
python test_model.py --use-sample --log-to-mlflow
```

---

## Summary

### Key Takeaways

1. **No "Best" Model** - It depends on your data and requirements
2. **Start Simple** - Try LogisticRegression or RandomForest first
3. **F1 Score Matters** - Especially for imbalanced data
4. **Always Cross-Validate** - Don't trust single splits
5. **Track Experiments** - Use MLflow to compare systematically
6. **Interpretability vs Accuracy** - Sometimes simple is better
7. **Consider Production Constraints** - Speed, memory, explainability

### Decision Framework

```
1. Define your problem clearly
2. Understand your data (size, balance, features)
3. Determine constraints (speed, interpretability, accuracy)
4. Select 2-3 candidate models
5. Compare using appropriate metrics
6. Tune the best performer
7. Validate on holdout set
8. Deploy with monitoring
```

---

**Last Updated:** 2025-10-24
**Version:** 1.0
**Maintained by:** MLflow Tutorial Project
