#!/usr/bin/env python3
"""
Hyperparameter tuning script with Grid Search and Random Search using MLflow tracking
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score, StratifiedKFold
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from utils.output_manager import OutputManager
import warnings
warnings.filterwarnings('ignore')

def prepare_data(test_size=0.2, random_state=42):
    """Load and prepare the iris dataset"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, iris.target_names

def get_hyperparameter_grids():
    """Define hyperparameter grids for different models"""
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        'SVM_RBF': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        },
        'SVM_Linear': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear']
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        },
        'DecisionTree': {
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
    return param_grids

def get_model_instances():
    """Get model instances for hyperparameter tuning"""
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42, probability=True),
        'SVM_Linear': SVC(kernel='linear', random_state=42, probability=True),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    return models

def perform_hyperparameter_tuning(model_name, model, param_grid, X_train, y_train,
                                search_type='grid', n_iter=50, cv_folds=5,
                                random_state=42, output_manager=None):
    """Perform hyperparameter tuning with either Grid Search or Random Search"""

    print(f"\n🔧 Tuning {model_name} using {search_type.upper()} search...")

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Choose search strategy
    if search_type.lower() == 'grid':
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
    else:  # random search
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )

    # Start MLflow run for this model's hyperparameter tuning
    with mlflow.start_run(run_name=f"{model_name}_{search_type}_search"):
        # Record start time
        start_time = time.time()

        # Perform the search
        search.fit(X_train, y_train)

        # Record end time
        tuning_time = time.time() - start_time

        # Log basic information
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("search_type", search_type)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("total_fits", len(search.cv_results_['mean_test_score']))

        if search_type.lower() == 'random':
            mlflow.log_param("n_iter", n_iter)

        # Log best parameters
        for param, value in search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)

        # Log performance metrics
        mlflow.log_metric("best_cv_score", search.best_score_)
        mlflow.log_metric("best_cv_std", search.cv_results_['std_test_score'][search.best_index_])
        mlflow.log_metric("tuning_time_seconds", tuning_time)

        # Log the best model
        mlflow.sklearn.log_model(
            sk_model=search.best_estimator_,
            artifact_path="best_model",
            signature=mlflow.models.infer_signature(X_train, search.best_estimator_.predict(X_train))
        )

        # Create and save detailed results
        results_df = pd.DataFrame(search.cv_results_)

        # Save hyperparameter tuning results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"outputs/results/hyperparameter_tuning/{model_name}_{search_type}_{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed CV results
        results_path = results_dir / "cv_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path))

        # Save best parameters as JSON
        best_params_path = results_dir / "best_parameters.json"
        with open(best_params_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'search_type': search_type,
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'tuning_time': tuning_time
            }, f, indent=2)
        mlflow.log_artifact(str(best_params_path))

        # Create visualization of hyperparameter search results
        create_tuning_visualization(search, model_name, search_type, results_dir)

        # Log the visualization
        viz_path = results_dir / "hyperparameter_search_results.png"
        if viz_path.exists():
            mlflow.log_artifact(str(viz_path))

        # Print results
        print(f"✅ {model_name} {search_type.upper()} search completed!")
        print(f"   Best CV Score: {search.best_score_:.4f}")
        print(f"   Best Parameters: {search.best_params_}")
        print(f"   Tuning Time: {tuning_time:.2f} seconds")
        print(f"   Results saved to: {results_dir}")

        return {
            'model_name': model_name,
            'search_type': search_type,
            'best_estimator': search.best_estimator_,
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'tuning_time': tuning_time,
            'run_id': mlflow.active_run().info.run_id,
            'results_dir': results_dir
        }

def create_tuning_visualization(search, model_name, search_type, output_dir):
    """Create visualization of hyperparameter tuning results"""

    results = search.cv_results_

    # Create subplots based on available data
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - {search_type.title()} Search Results', fontsize=16)

    # 1. Score distribution
    axes[0, 0].hist(results['mean_test_score'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(search.best_score_, color='red', linestyle='--',
                      label=f'Best Score: {search.best_score_:.4f}')
    axes[0, 0].set_xlabel('Cross-Validation Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('CV Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Score vs Rank
    ranks = results['rank_test_score']
    scores = results['mean_test_score']
    axes[0, 1].scatter(ranks, scores, alpha=0.6)
    axes[0, 1].set_xlabel('Parameter Combination Rank')
    axes[0, 1].set_ylabel('CV Score')
    axes[0, 1].set_title('Score vs Parameter Rank')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Training vs Validation scores (if available)
    if 'mean_train_score' in results:
        train_scores = results['mean_train_score']
        val_scores = results['mean_test_score']
        axes[1, 0].scatter(train_scores, val_scores, alpha=0.6)
        axes[1, 0].plot([min(train_scores), max(train_scores)],
                       [min(train_scores), max(train_scores)], 'r--', alpha=0.8)
        axes[1, 0].set_xlabel('Training Score')
        axes[1, 0].set_ylabel('Validation Score')
        axes[1, 0].set_title('Training vs Validation Scores')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Training scores not available',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training vs Validation Scores')

    # 4. Best parameters visualization (text)
    best_params_text = "Best Parameters:\n\n"
    for param, value in search.best_params_.items():
        best_params_text += f"{param}: {value}\n"

    best_params_text += f"\nBest CV Score: {search.best_score_:.4f}"
    best_params_text += f"\nStd: ±{results['std_test_score'][search.best_index_]:.4f}"

    axes[1, 1].text(0.1, 0.9, best_params_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Best Configuration')

    plt.tight_layout()

    # Save the plot
    viz_path = output_dir / "hyperparameter_search_results.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_best_models(tuning_results, X_test, y_test, class_names):
    """Evaluate the best models from hyperparameter tuning"""

    print(f"\n📊 Evaluating best models on test set...")

    evaluation_results = []

    for result in tuning_results:
        model_name = result['model_name']
        search_type = result['search_type']
        best_model = result['best_estimator']

        # Predict on test set
        y_pred = best_model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        }

        evaluation_results.append({
            'model_name': model_name,
            'search_type': search_type,
            'cv_score': result['best_score'],
            'test_accuracy': metrics['accuracy'],
            'test_f1_macro': metrics['f1_macro'],
            'tuning_time': result['tuning_time'],
            'run_id': result['run_id']
        })

        print(f"✅ {model_name} ({search_type}): Test Accuracy = {metrics['accuracy']:.4f}")

    return evaluation_results

def create_comparison_visualization(evaluation_results, output_dir):
    """Create comparison visualization of all tuned models"""

    df = pd.DataFrame(evaluation_results)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Tuning Results Comparison', fontsize=16)

    # Model labels for better readability
    df['model_label'] = df['model_name'] + '_' + df['search_type']

    # 1. CV Score vs Test Accuracy
    axes[0, 0].scatter(df['cv_score'], df['test_accuracy'], s=100, alpha=0.7)
    for i, row in df.iterrows():
        axes[0, 0].annotate(row['model_label'], (row['cv_score'], row['test_accuracy']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('CV Score')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('CV Score vs Test Accuracy')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Test Accuracy comparison
    axes[0, 1].bar(range(len(df)), df['test_accuracy'])
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('Test Accuracy Comparison')
    axes[0, 1].set_xticks(range(len(df)))
    axes[0, 1].set_xticklabels(df['model_label'], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. F1-Score comparison
    axes[1, 0].bar(range(len(df)), df['test_f1_macro'])
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('F1-Score (Macro)')
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_xticks(range(len(df)))
    axes[1, 0].set_xticklabels(df['model_label'], rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Tuning Time comparison
    axes[1, 1].bar(range(len(df)), df['tuning_time'])
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Tuning Time (seconds)')
    axes[1, 1].set_title('Hyperparameter Tuning Time')
    axes[1, 1].set_xticks(range(len(df)))
    axes[1, 1].set_xticklabels(df['model_label'], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    viz_path = output_dir / "hyperparameter_tuning_comparison.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    return viz_path

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with MLflow tracking")
    parser.add_argument("--models", nargs='+',
                       help="Specific models to tune (default: all available)")
    parser.add_argument("--search-type", choices=['grid', 'random', 'both'],
                       default='both', help="Search strategy (default: both)")
    parser.add_argument("--n-iter", type=int, default=50,
                       help="Number of iterations for random search (default: 50)")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of CV folds (default: 5)")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Base output directory")

    args = parser.parse_args()

    # Initialize output manager
    output_manager = OutputManager(args.output_dir)
    output_manager.logger.info("Starting Hyperparameter Tuning with MLflow")

    print("🎯 MLflow Hyperparameter Tuning - Iris Classification")
    print("=" * 60)

    # Set MLflow experiment
    mlflow.set_experiment("iris-hyperparameter-tuning")

    # Prepare data
    print("📊 Preparing data...")
    X_train, X_test, y_train, y_test, class_names = prepare_data(
        test_size=args.test_size,
        random_state=args.random_state
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {list(X_train.columns)}")
    print(f"Classes: {class_names}")

    # Get models and parameter grids
    models = get_model_instances()
    param_grids = get_hyperparameter_grids()

    # Filter models if specified
    if args.models:
        available_models = list(models.keys())
        requested_models = [m for m in args.models if m in available_models]
        if not requested_models:
            print(f"❌ No valid models found. Available: {available_models}")
            return
        models = {k: v for k, v in models.items() if k in requested_models}
        param_grids = {k: v for k, v in param_grids.items() if k in requested_models}
        print(f"🎯 Tuning specific models: {list(models.keys())}")
    else:
        print(f"🎯 Tuning all models: {list(models.keys())}")

    # Perform hyperparameter tuning
    tuning_results = []
    search_types = []

    if args.search_type in ['grid', 'both']:
        search_types.append('grid')
    if args.search_type in ['random', 'both']:
        search_types.append('random')

    total_experiments = len(models) * len(search_types)
    current_exp = 0

    for model_name, model in models.items():
        for search_type in search_types:
            current_exp += 1
            print(f"\n📈 Progress: {current_exp}/{total_experiments}")

            result = perform_hyperparameter_tuning(
                model_name=model_name,
                model=model,
                param_grid=param_grids[model_name],
                X_train=X_train,
                y_train=y_train,
                search_type=search_type,
                n_iter=args.n_iter,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                output_manager=output_manager
            )
            tuning_results.append(result)

    # Evaluate best models on test set
    evaluation_results = evaluate_best_models(tuning_results, X_test, y_test, class_names)

    # Create comparison visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = Path(f"outputs/results/hyperparameter_tuning/comparison_{timestamp}")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    viz_path = create_comparison_visualization(evaluation_results, comparison_dir)

    # Save evaluation results
    eval_df = pd.DataFrame(evaluation_results)
    results_path = comparison_dir / "tuning_evaluation_results.csv"
    eval_df.to_csv(results_path, index=False)

    # Log summary to MLflow
    with mlflow.start_run(run_name="Hyperparameter_Tuning_Summary"):
        mlflow.log_param("total_models_tuned", len(tuning_results))
        mlflow.log_param("search_types", search_types)
        mlflow.log_param("cv_folds", args.cv_folds)

        # Find best overall model
        best_model = eval_df.loc[eval_df['test_accuracy'].idxmax()]
        mlflow.log_param("best_overall_model", best_model['model_name'])
        mlflow.log_param("best_search_type", best_model['search_type'])
        mlflow.log_metric("best_test_accuracy", best_model['test_accuracy'])
        mlflow.log_metric("best_cv_score", best_model['cv_score'])

        # Log artifacts
        mlflow.log_artifact(str(viz_path))
        mlflow.log_artifact(str(results_path))

    # Print final summary
    print(f"\n🏆 HYPERPARAMETER TUNING SUMMARY")
    print("=" * 50)
    print(f"📊 Total experiments: {len(tuning_results)}")
    print(f"🥇 Best model: {best_model['model_name']} ({best_model['search_type']})")
    print(f"🎯 Best test accuracy: {best_model['test_accuracy']:.4f}")
    print(f"📈 Best CV score: {best_model['cv_score']:.4f}")
    print(f"\n📋 Detailed Results:")
    print(eval_df[['model_name', 'search_type', 'cv_score', 'test_accuracy', 'tuning_time']].round(4))

    print(f"\n✅ Results saved to: {comparison_dir}")
    print(f"📊 View results in MLflow UI: http://localhost:5001")
    print(f"🔍 Experiment: iris-hyperparameter-tuning")

if __name__ == "__main__":
    main()