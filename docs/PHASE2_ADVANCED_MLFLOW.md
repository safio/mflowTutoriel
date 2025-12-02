# Phase 2: Advanced MLflow Features

This guide covers the advanced MLflow features implemented in Phase 2, including Model Registry, Model Serving, and Automated Model Validation.

## Table of Contents

1. [MLflow Model Registry](#mlflow-model-registry)
2. [Model Serving & Deployment](#model-serving--deployment)
3. [Automated Model Validation](#automated-model-validation)
4. [Model Performance Monitoring](#model-performance-monitoring)
5. [Complete Workflow Examples](#complete-workflow-examples)

---

## MLflow Model Registry

The Model Registry provides centralized model versioning, stage transitions, and lifecycle management.

### Core Components

#### 1. MLflow Registry Manager (`mlflow_registry_manager.py`)

Comprehensive Python API for Model Registry operations:

```python
from mlflow_tutorial.experiments.mlflow_registry_manager import MLflowRegistryManager

# Initialize manager
registry = MLflowRegistryManager()

# Register a model from a run
version = registry.register_model(
    run_id="abc123...",
    model_name="iris-classifier",
    tags={"framework": "sklearn", "dataset": "iris"},
    description="Random Forest model trained on iris dataset"
)

# Transition to Production
registry.transition_model_stage(
    model_name="iris-classifier",
    version="1",
    stage="Production"
)

# Compare model versions
comparison = registry.compare_model_versions(
    model_name="iris-classifier",
    version1="1",
    version2="2",
    metric_keys=["accuracy", "f1_score"]
)

# Auto-promote based on metrics
promoted = registry.auto_promote_model(
    model_name="iris-classifier",
    staging_version="2",
    metric_name="accuracy",
    metric_threshold=0.95
)
```

#### 2. Model Registration Script (`register_model.py`)

Command-line tool for managing registered models:

```bash
# Register a model from a run
python register_model.py \
  --run-id <run_id> \
  --model-name iris-classifier \
  --stage Staging \
  --tags framework=sklearn dataset=iris \
  --description "Initial model version"

# List all registered models
python register_model.py --list-models

# Get model details
python register_model.py \
  --model-name iris-classifier \
  --details

# Promote model to Production
python register_model.py \
  --model-name iris-classifier \
  --version 1 \
  --promote \
  --from-stage Staging \
  --to-stage Production

# Auto-promote based on metrics
python register_model.py \
  --model-name iris-classifier \
  --version 2 \
  --auto-promote \
  --metric accuracy \
  --threshold 0.95

# Compare two versions
python register_model.py \
  --model-name iris-classifier \
  --version 1 \
  --compare-version 2 \
  --compare

# Rollback to previous version
python register_model.py \
  --model-name iris-classifier \
  --version 1 \
  --rollback \
  --yes
```

### Model Lifecycle Stages

Models progress through the following stages:

1. **None** - Initial registration
2. **Staging** - Testing and validation
3. **Production** - Active deployment
4. **Archived** - Deprecated or replaced

---

## Model Serving & Deployment

### 1. Model Serving with REST API (`model_serving.py`)

Serve models via Flask REST API:

```bash
# Serve Production model
python model_serving.py \
  --model-name iris-classifier \
  --stage Production \
  --host 0.0.0.0 \
  --port 5002

# Serve specific version
python model_serving.py \
  --model-name iris-classifier \
  --version 2 \
  --port 5002

# Serve model from run ID
python model_serving.py \
  --run-id <run_id> \
  --port 5002
```

#### API Endpoints

**Health Check:**
```bash
curl http://localhost:5002/health
```

**Model Info:**
```bash
curl http://localhost:5002/model/info
```

**Single Prediction:**
```bash
curl -X POST http://localhost:5002/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "features": [5.1, 3.5, 1.4, 0.2]
  }'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5002/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "instances": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 2.9, 4.3, 1.3],
      [7.3, 2.9, 6.3, 1.8]
    ]
  }'
```

### 2. Batch Predictions (`batch_prediction.py`)

Process large datasets efficiently:

```bash
# Predict using Production model
python batch_prediction.py \
  --model-name iris-classifier \
  --input data/test_data.csv \
  --output results/predictions.csv \
  --log-to-mlflow

# Predict using specific version
python batch_prediction.py \
  --model-name iris-classifier \
  --version 2 \
  --input data/test_data.csv \
  --show-sample 10

# Use model from run ID
python batch_prediction.py \
  --run-id <run_id> \
  --input data/test_data.csv \
  --output predictions.csv
```

Features:
- Automatic output file management
- MLflow logging of prediction batches
- Prediction distribution analysis
- Timestamp tracking

### 3. Real-time Predictions (`realtime_prediction.py`)

Interactive prediction demonstrations:

```bash
# Interactive mode
python realtime_prediction.py \
  --model-name iris-classifier

# Demo mode with sample data
python realtime_prediction.py \
  --model-name iris-classifier \
  --demo

# Single prediction from command line
python realtime_prediction.py \
  --model-name iris-classifier \
  --predict 5.1,3.5,1.4,0.2

# Use Staging model
python realtime_prediction.py \
  --model-name iris-classifier \
  --stage Staging
```

---

## Automated Model Validation

### Model Validation & A/B Testing (`model_validation.py`)

Comprehensive validation framework with statistical significance testing:

```bash
# Compare Production vs Staging
python model_validation.py \
  --model-name iris-classifier \
  --champion-stage Production \
  --challenger-stage Staging

# Compare specific versions
python model_validation.py \
  --model-name iris-classifier \
  --champion-version 1 \
  --challenger-version 2

# Use custom test data
python model_validation.py \
  --model-name iris-classifier \
  --champion-version 1 \
  --challenger-version 2 \
  --test-data data/validation_set.csv

# Auto-promote if challenger wins
python model_validation.py \
  --model-name iris-classifier \
  --champion-stage Production \
  --challenger-stage Staging \
  --auto-promote \
  --confidence 0.95
```

#### Features

**Statistical Testing:**
- McNemar's test for paired predictions
- Confidence level configuration (default: 95%)
- P-value and chi-square statistics

**Performance Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Per-class performance analysis
- Confusion matrices for both models

**Visualization:**
- Side-by-side metric comparison
- Percentage change visualization
- Confusion matrix heatmaps
- Automated plot generation

**Decision Recommendations:**
- Strong promotion recommendation (75%+ metrics improved)
- Conditional recommendation (50%+ metrics improved)
- Keep champion (champion performs better)
- Insufficient evidence (not statistically significant)

---

## Model Performance Monitoring

### Performance Monitoring (`model_monitoring.py`)

Track model performance over time and detect degradation:

```bash
# Generate monitoring report
python model_monitoring.py \
  --model-name iris-classifier \
  --report \
  --days 30

# Track new predictions
python model_monitoring.py \
  --model-name iris-classifier \
  --track-predictions \
  --input data/new_data.csv \
  --labels data/true_labels.csv

# Check for performance degradation
python model_monitoring.py \
  --model-name iris-classifier \
  --check-degradation \
  --threshold 0.05

# Plot performance over time
python model_monitoring.py \
  --model-name iris-classifier \
  --plot \
  --days 30 \
  --metrics accuracy f1_score precision recall
```

#### Features

**Performance Tracking:**
- Continuous monitoring of prediction batches
- Historical performance analysis
- Metric trends over time

**Degradation Detection:**
- Configurable thresholds
- Window-based comparison (recent vs baseline)
- Automatic alert recommendations

**Visualization:**
- Performance metrics over time
- Trend lines and patterns
- Multi-metric dashboards

**Reporting:**
- Comprehensive monitoring summaries
- Summary statistics (mean, std, min, max)
- Degradation analysis
- Retraining recommendations

---

## Complete Workflow Examples

### Example 1: Training to Production Deployment

```bash
# 1. Train multiple models
python compare_models_structured.py \
  --models RandomForest SVM_Linear LogisticRegression

# 2. Register best model to MLflow Registry
# (Get run_id from MLflow UI or training output)
python register_model.py \
  --run-id <best_model_run_id> \
  --model-name iris-classifier \
  --stage Staging \
  --description "Initial production candidate"

# 3. Validate the model
python model_validation.py \
  --model-name iris-classifier \
  --challenger-stage Staging \
  --champion-version baseline \
  --auto-promote \
  --confidence 0.95

# 4. Serve the Production model
python model_serving.py \
  --model-name iris-classifier \
  --stage Production \
  --port 5002

# 5. Start monitoring
python model_monitoring.py \
  --model-name iris-classifier \
  --report
```

### Example 2: Model Update and A/B Testing

```bash
# 1. Train new model with hyperparameter tuning
python hyperparameter_tuning.py \
  --models RandomForest \
  --search-type grid

# 2. Register new version to Staging
python register_model.py \
  --run-id <new_model_run_id> \
  --model-name iris-classifier \
  --stage Staging \
  --description "Hyperparameter optimized version"

# 3. A/B test: Production vs Staging
python model_validation.py \
  --model-name iris-classifier \
  --champion-stage Production \
  --challenger-stage Staging \
  --save-plot

# 4. If challenger wins, promote manually or auto-promote
python register_model.py \
  --model-name iris-classifier \
  --version 2 \
  --promote \
  --to-stage Production

# 5. Monitor new production model
python model_monitoring.py \
  --model-name iris-classifier \
  --plot \
  --days 7
```

### Example 3: Batch Prediction Pipeline

```bash
# 1. Prepare test data
# (create test_data.csv with features)

# 2. Run batch predictions
python batch_prediction.py \
  --model-name iris-classifier \
  --stage Production \
  --input test_data.csv \
  --output predictions.csv \
  --log-to-mlflow \
  --show-sample 5

# 3. Track predictions for monitoring
python model_monitoring.py \
  --model-name iris-classifier \
  --track-predictions \
  --input test_data.csv \
  --labels true_labels.csv

# 4. Check for degradation
python model_monitoring.py \
  --model-name iris-classifier \
  --check-degradation
```

### Example 4: Model Rollback Scenario

```bash
# 1. Detect performance issue
python model_monitoring.py \
  --model-name iris-classifier \
  --check-degradation \
  --threshold 0.05

# 2. Compare current Production with previous version
python model_validation.py \
  --model-name iris-classifier \
  --champion-version 3 \
  --challenger-version 2

# 3. Rollback to previous version if needed
python register_model.py \
  --model-name iris-classifier \
  --version 2 \
  --rollback \
  --to-stage Production \
  --yes

# 4. Verify rollback
python register_model.py \
  --model-name iris-classifier \
  --details
```

---

## Best Practices

### Model Registry

1. **Naming Conventions:**
   - Use descriptive model names (e.g., `iris-classifier-rf`, `iris-classifier-svm`)
   - Include model type or algorithm in name
   - Use consistent naming across projects

2. **Version Management:**
   - Always test in Staging before Production
   - Archive old Production models instead of deleting
   - Document changes in model descriptions

3. **Tags:**
   - Tag models with framework, dataset, and training date
   - Use tags for experiment tracking and filtering
   - Include performance metrics as tags

### Model Serving

1. **API Design:**
   - Implement health checks for monitoring
   - Validate input data before prediction
   - Return meaningful error messages
   - Include model version in responses

2. **Performance:**
   - Use batch endpoints for multiple predictions
   - Implement caching for frequently requested predictions
   - Monitor API response times

3. **Security:**
   - Run on localhost for development
   - Use authentication for production deployments
   - Validate and sanitize inputs

### Model Validation

1. **Testing:**
   - Use hold-out test sets (not training data)
   - Ensure test data is representative
   - Test on recent/production-like data

2. **Statistical Rigor:**
   - Use appropriate confidence levels (95% recommended)
   - Consider sample size for significance
   - Don't rely solely on automated decisions

3. **Decision Making:**
   - Review detailed metrics, not just overall accuracy
   - Consider business impact, not just statistical significance
   - Test thoroughly before production deployment

### Monitoring

1. **Data Collection:**
   - Log all predictions with timestamps
   - Track both predictions and actuals when available
   - Record metadata (data source, model version, etc.)

2. **Alerting:**
   - Set appropriate degradation thresholds
   - Monitor multiple metrics, not just accuracy
   - Implement automated alerts for significant degradation

3. **Maintenance:**
   - Review monitoring reports regularly
   - Investigate degradation causes
   - Retrain models when performance drops

---

## Troubleshooting

### Common Issues

**1. Model Not Found:**
```
Error: Model 'iris-classifier' not found
```
**Solution:** Ensure model is registered. List models with:
```bash
python register_model.py --list-models
```

**2. No Model in Stage:**
```
Error: No model version in 'Production' stage
```
**Solution:** Transition model to correct stage:
```bash
python register_model.py --model-name iris-classifier --version 1 --promote
```

**3. Serving Port Already in Use:**
```
Error: Address already in use
```
**Solution:** Use different port or stop existing server:
```bash
python model_serving.py --model-name iris-classifier --port 5003
```

**4. Insufficient Data for Monitoring:**
```
Insufficient data for degradation analysis
```
**Solution:** Track more prediction batches before analysis:
```bash
python model_monitoring.py --model-name iris-classifier --track-predictions --input data.csv
```

---

## Next Steps

After completing Phase 2, you can:

1. **Explore Phase 3:** Different datasets and use cases (regression, time series, NLP)
2. **Production Deployment:** Containerize with Docker, deploy to cloud platforms
3. **CI/CD Integration:** Automate model training and deployment pipelines
4. **Advanced Monitoring:** Implement data drift detection and automated retraining

---

## Resources

- [MLflow Model Registry Documentation](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Model Serving Guide](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)
- [Statistical Testing for ML Models](https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/)
- [A/B Testing Best Practices](https://www.optimizely.com/optimization-glossary/ab-testing/)

---

**Congratulations on completing Phase 2!** 🎉

You now have a production-ready MLflow setup with model registry, serving, validation, and monitoring capabilities.
