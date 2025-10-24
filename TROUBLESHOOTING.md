# MLflow Tutorial - Troubleshooting Guide

This document contains solutions to common issues you may encounter when working with this MLflow tutorial project.

## Table of Contents
- [MLflow Experiment Errors](#mlflow-experiment-errors)
- [MLflow UI Issues](#mlflow-ui-issues)
- [Data and Model Issues](#data-and-model-issues)

---

## MLflow Experiment Errors

### Issue: "Could not create run under non-active experiment"

**Error Message:**
```
mlflow.exceptions.MlflowException: Could not create run under non-active experiment with ID [experiment_id]
```

**Cause:**
This error occurs when experiments have been deleted from the MLflow UI but not permanently removed from the tracking store. When an experiment is deleted via the MLflow UI, it moves to a "deleted" lifecycle stage rather than being completely removed. Scripts attempting to use `mlflow.set_experiment()` with a deleted experiment name will encounter this error.

**Common Scenarios:**
1. Someone deleted experiments from the MLflow UI to "clean up"
2. Experiments were deleted during testing/development
3. The `mlruns/.trash` directory contains deleted experiments
4. Team members sharing the same MLflow tracking directory

**Impact:**
- All scripts using `mlflow.set_experiment()` will fail
- Cannot create new runs or track experiments
- Blocks all model training and comparison workflows

**Solution:**

**Option 1: Quick Fix (Recommended)**
Remove the MLflow trash directory to permanently delete old experiments:

```bash
# Stop any running MLflow UI servers first
rm -rf mlruns/.trash

# Test that it works
python train.py --n_estimators 50
```

**Option 2: Use the Fix Script**
Run the provided utility script:

```bash
python fix_mlflow_experiments.py
```

This script will:
1. List all deleted experiments
2. Provide instructions for cleanup
3. Verify the fix

**Option 3: Complete Reset (Nuclear Option)**
If you want to start completely fresh with no history:

```bash
# CAUTION: This deletes ALL experiment history!
rm -rf mlruns/
python train.py --n_estimators 50  # Creates fresh mlruns directory
```

**Prevention:**
1. **Don't delete experiments from the MLflow UI** unless you understand the implications
2. Use experiment names with version suffixes if you need to start fresh (e.g., `iris-classification-v2`)
3. Add `mlruns/.trash` to `.gitignore` to prevent committing deleted experiments
4. Document experiment lifecycle policies for your team

**Verification:**
After applying the fix, verify your experiments are working:

```bash
# Check active experiments
python -c "import mlflow; client = mlflow.MlflowClient(); \
experiments = client.search_experiments(view_type=1); \
print('Active Experiments:'); \
[print(f'  - {exp.name} (ID: {exp.experiment_id})') for exp in experiments]"

# Test a training run
python train.py --n_estimators 50
```

---

## MLflow UI Issues

### Issue: MLflow UI shows old/deleted experiments

**Solution:**
The MLflow UI caches experiment data. To see a clean state:

```bash
# Stop the MLflow UI
# Press Ctrl+C in the terminal running the UI

# Clean up deleted experiments
rm -rf mlruns/.trash

# Restart the UI
python -m mlflow ui --host 0.0.0.0 --port 5001
```

### Issue: Port already in use

**Error Message:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find and kill the process using port 5001
lsof -ti:5001 | xargs kill -9

# Or use a different port
python -m mlflow ui --host 0.0.0.0 --port 5002
```

---

## Data and Model Issues

### Issue: Import errors with sklearn

**Error Message:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
Ensure your virtual environment is activated and dependencies are installed:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Model artifacts not found

**Solution:**
Ensure you're running scripts from the project root directory:

```bash
cd /path/to/mflowTutoriel
python train.py
```

---

## Best Practices to Avoid Issues

### 1. MLflow Experiment Management
- **Never delete experiments from the UI** in a shared/production environment
- Use versioned experiment names for different phases (e.g., `iris-classification-dev`, `iris-classification-prod`)
- Document your experiment naming conventions

### 2. Git and Version Control
- Keep `mlruns/.trash` in `.gitignore`
- Don't commit the entire `mlruns` directory unless necessary
- Use `.gitkeep` files to maintain directory structure

### 3. Team Collaboration
- Share experiment tracking servers rather than local `mlruns` directories
- Document the MLflow tracking URI in your README
- Establish experiment lifecycle policies

### 4. Development Workflow
```bash
# Before starting work
source venv/bin/activate
python -m mlflow ui --host 0.0.0.0 --port 5001 &

# After cleaning up experiments
rm -rf mlruns/.trash
git status  # Check what changed

# Regular workflow
python train.py
python compare_models_structured.py
```

---

## Getting Help

If you encounter issues not covered here:

1. Check the [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
2. Review the project's `CLAUDE.md` for project-specific guidance
3. Check MLflow logs in `outputs/logs/` directory
4. Verify your Python environment and dependencies

---

## Quick Reference Commands

```bash
# List all experiments (including deleted)
python -c "import mlflow; client = mlflow.MlflowClient(); \
[print(f'{exp.name}: {exp.lifecycle_stage}') for exp in client.search_experiments(view_type=3)]"

# Clean deleted experiments
rm -rf mlruns/.trash

# Start fresh MLflow UI
python -m mlflow ui --host 0.0.0.0 --port 5001

# Test basic training
python train.py --n_estimators 50

# Check environment
source venv/bin/activate && pip list | grep mlflow
```

---

**Last Updated:** 2025-10-24
**Maintainer:** Project Team
