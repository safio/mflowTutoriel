#!/usr/bin/env python3
"""
Comprehensive K-fold Cross-Validation script with MLflow tracking
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, cross_validate,
    LeaveOneOut, ShuffleSplit, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

def prepare_data():
    """Load and prepare the iris dataset"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    return X, y, iris.target_names

def get_model_configs():
    """Define all models for cross-validation"""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42, probability=True),
        'SVM_Linear': SVC(kernel='linear', random_state=42, probability=True),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    return models

def get_cv_strategies(n_splits=5, random_state=42):
    """Define different cross-validation strategies"""
    strategies = {
        'KFold': KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        'StratifiedKFold': StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        'ShuffleSplit': ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state),
        'LeaveOneOut': LeaveOneOut() if len(load_iris().data) <= 200 else None  # Only for small datasets
    }

    # Remove None strategies
    strategies = {k: v for k, v in strategies.items() if v is not None}
    return strategies

def perform_cross_validation(model_name, model, X, y, cv_strategy_name, cv_strategy,
                           scoring_metrics, class_names, output_manager=None):
    """Perform cross-validation with detailed tracking"""

    print(f"\n🔄 Cross-validating {model_name} using {cv_strategy_name}...")

    with mlflow.start_run(run_name=f"{model_name}_{cv_strategy_name}_CV"):
        start_time = time.time()

        # Log basic parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("cv_strategy", cv_strategy_name)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", len(X.columns))
        mlflow.log_param("n_classes", len(class_names))

        # Get number of splits for this CV strategy
        try:
            n_splits = cv_strategy.get_n_splits(X, y)
            mlflow.log_param("n_splits", n_splits)
        except:
            n_splits = getattr(cv_strategy, 'n_splits', 'variable')
            mlflow.log_param("n_splits", str(n_splits))

        # Perform cross-validation with multiple metrics
        cv_results = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=cv_strategy,
            scoring=scoring_metrics,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1
        )

        cv_time = time.time() - start_time
        mlflow.log_metric("cv_time_seconds", cv_time)

        # Calculate and log statistics for each metric
        results_summary = {}

        for metric in scoring_metrics:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']

            # Test scores statistics
            test_mean = np.mean(test_scores)
            test_std = np.std(test_scores)
            test_min = np.min(test_scores)
            test_max = np.max(test_scores)

            # Train scores statistics
            train_mean = np.mean(train_scores)
            train_std = np.std(train_scores)

            # Log to MLflow
            mlflow.log_metric(f"{metric}_test_mean", test_mean)
            mlflow.log_metric(f"{metric}_test_std", test_std)
            mlflow.log_metric(f"{metric}_test_min", test_min)
            mlflow.log_metric(f"{metric}_test_max", test_max)
            mlflow.log_metric(f"{metric}_train_mean", train_mean)
            mlflow.log_metric(f"{metric}_train_std", train_std)

            # Store for summary
            results_summary[metric] = {
                'test_scores': test_scores,
                'train_scores': train_scores,
                'test_mean': test_mean,
                'test_std': test_std,
                'train_mean': train_mean,
                'train_std': train_std
            }

        # Calculate overfitting metrics
        accuracy_gap = results_summary['accuracy']['train_mean'] - results_summary['accuracy']['test_mean']
        mlflow.log_metric("overfitting_gap", accuracy_gap)

        # Create detailed results DataFrame
        cv_details = []
        for fold_idx, (train_score, test_score) in enumerate(zip(
            cv_results['train_accuracy'], cv_results['test_accuracy']
        )):
            cv_details.append({
                'fold': fold_idx + 1,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'gap': train_score - test_score
            })

        cv_df = pd.DataFrame(cv_details)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"outputs/results/cross_validation/{model_name}_{cv_strategy_name}_{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save CV details
        cv_details_path = results_dir / "cv_fold_details.csv"
        cv_df.to_csv(cv_details_path, index=False)
        mlflow.log_artifact(str(cv_details_path))

        # Save full CV results
        cv_results_path = results_dir / "cv_complete_results.json"
        # Convert numpy arrays to lists for JSON serialization
        cv_results_serializable = {}
        for key, value in cv_results.items():
            if key != 'estimator':  # Skip estimator objects
                if isinstance(value, np.ndarray):
                    cv_results_serializable[key] = value.tolist()
                else:
                    cv_results_serializable[key] = value

        with open(cv_results_path, 'w') as f:
            json.dump(cv_results_serializable, f, indent=2)
        mlflow.log_artifact(str(cv_results_path))

        # Save summary statistics
        summary_path = results_dir / "cv_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'cv_strategy': cv_strategy_name,
                'n_splits': str(n_splits),
                'cv_time': cv_time,
                'metrics_summary': {
                    metric: {
                        'test_mean': float(stats['test_mean']),
                        'test_std': float(stats['test_std']),
                        'train_mean': float(stats['train_mean']),
                        'train_std': float(stats['train_std'])
                    }
                    for metric, stats in results_summary.items()
                },
                'overfitting_gap': float(accuracy_gap)
            }, f, indent=2)
        mlflow.log_artifact(str(summary_path))

        # Create visualizations
        create_cv_visualizations(model_name, cv_strategy_name, results_summary, cv_df, results_dir)

        # Log visualizations
        for viz_file in results_dir.glob("*.png"):
            mlflow.log_artifact(str(viz_file))

        # Log the best estimator (from best fold)
        best_fold_idx = np.argmax(cv_results['test_accuracy'])
        best_estimator = cv_results['estimator'][best_fold_idx]

        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="best_fold_model",
            signature=mlflow.models.infer_signature(X, best_estimator.predict(X))
        )

        # Print results
        print(f"✅ {model_name} ({cv_strategy_name}) Results:")
        print(f"   Accuracy: {results_summary['accuracy']['test_mean']:.4f} ± {results_summary['accuracy']['test_std']:.4f}")
        print(f"   F1-Score: {results_summary['f1_macro']['test_mean']:.4f} ± {results_summary['f1_macro']['test_std']:.4f}")
        print(f"   Overfitting Gap: {accuracy_gap:.4f}")
        print(f"   CV Time: {cv_time:.2f}s")

        return {
            'model_name': model_name,
            'cv_strategy': cv_strategy_name,
            'run_id': mlflow.active_run().info.run_id,
            'results_summary': results_summary,
            'cv_time': cv_time,
            'overfitting_gap': accuracy_gap,
            'n_splits': str(n_splits),
            'results_dir': results_dir
        }

def create_cv_visualizations(model_name, cv_strategy_name, results_summary, cv_df, output_dir):
    """Create comprehensive visualizations for cross-validation results"""

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - {cv_strategy_name} Cross-Validation Results', fontsize=16)

    # 1. Accuracy across folds
    axes[0, 0].plot(cv_df['fold'], cv_df['train_accuracy'], 'o-', label='Train', linewidth=2, markersize=8)
    axes[0, 0].plot(cv_df['fold'], cv_df['test_accuracy'], 's-', label='Test', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Across Folds')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.8, 1.0])

    # 2. Overfitting analysis (train vs test scores)
    axes[0, 1].scatter(cv_df['train_accuracy'], cv_df['test_accuracy'], s=100, alpha=0.7, c='blue')
    axes[0, 1].plot([0.8, 1.0], [0.8, 1.0], 'r--', alpha=0.8, label='Perfect Fit')
    axes[0, 1].set_xlabel('Training Accuracy')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('Overfitting Analysis')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Score distribution across folds
    metrics_to_plot = ['accuracy', 'f1_macro']
    x_pos = np.arange(len(metrics_to_plot))

    test_means = [results_summary[metric]['test_mean'] for metric in metrics_to_plot]
    test_stds = [results_summary[metric]['test_std'] for metric in metrics_to_plot]
    train_means = [results_summary[metric]['train_mean'] for metric in metrics_to_plot]
    train_stds = [results_summary[metric]['train_std'] for metric in metrics_to_plot]

    width = 0.35
    axes[1, 0].bar(x_pos - width/2, test_means, width, yerr=test_stds,
                   label='Test', alpha=0.8, capsize=5)
    axes[1, 0].bar(x_pos + width/2, train_means, width, yerr=train_stds,
                   label='Train', alpha=0.8, capsize=5)

    axes[1, 0].set_xlabel('Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Mean Scores with Standard Deviation')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Statistical summary box
    summary_text = f"Cross-Validation Summary\n"
    summary_text += f"{'='*30}\n\n"
    summary_text += f"Model: {model_name}\n"
    summary_text += f"CV Strategy: {cv_strategy_name}\n"
    summary_text += f"Number of Folds: {len(cv_df)}\n\n"

    for metric, stats in results_summary.items():
        if metric in ['accuracy', 'f1_macro']:
            summary_text += f"{metric.replace('_', ' ').title()}:\n"
            summary_text += f"  Test:  {stats['test_mean']:.4f} ± {stats['test_std']:.4f}\n"
            summary_text += f"  Train: {stats['train_mean']:.4f} ± {stats['train_std']:.4f}\n\n"

    overfitting_gap = results_summary['accuracy']['train_mean'] - results_summary['accuracy']['test_mean']
    summary_text += f"Overfitting Gap: {overfitting_gap:.4f}\n"

    if overfitting_gap < 0.05:
        summary_text += "Status: ✅ Good generalization"
    elif overfitting_gap < 0.10:
        summary_text += "Status: ⚠️ Mild overfitting"
    else:
        summary_text += "Status: ❌ Significant overfitting"

    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistical Summary')

    plt.tight_layout()

    # Save the main plot
    main_viz_path = output_dir / f"cv_analysis_{model_name}_{cv_strategy_name}.png"
    plt.savefig(main_viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create additional fold-by-fold boxplot
    create_fold_boxplot(results_summary, output_dir, model_name, cv_strategy_name)

def create_fold_boxplot(results_summary, output_dir, model_name, cv_strategy_name):
    """Create boxplot showing score distribution across folds"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Prepare data for boxplot
    metrics_data = []
    labels = []

    for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
        if metric in results_summary:
            test_scores = results_summary[metric]['test_scores']
            metrics_data.append(test_scores)
            labels.append(metric.replace('_', ' ').title())

    # Create boxplot
    bp = ax.boxplot(metrics_data, labels=labels, patch_artist=True)

    # Customize colors
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} - {cv_strategy_name}: Score Distribution Across Folds')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.8, 1.0])

    plt.xticks(rotation=45)
    plt.tight_layout()

    boxplot_path = output_dir / f"cv_boxplot_{model_name}_{cv_strategy_name}.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_visualization(all_results, output_dir):
    """Create comparison visualization across all models and CV strategies"""

    # Convert results to DataFrame
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'model': result['model_name'],
            'cv_strategy': result['cv_strategy'],
            'accuracy_mean': result['results_summary']['accuracy']['test_mean'],
            'accuracy_std': result['results_summary']['accuracy']['test_std'],
            'f1_mean': result['results_summary']['f1_macro']['test_mean'],
            'f1_std': result['results_summary']['f1_macro']['test_std'],
            'overfitting_gap': result['overfitting_gap'],
            'cv_time': result['cv_time']
        })

    df = pd.DataFrame(comparison_data)
    df['model_cv'] = df['model'] + '_' + df['cv_strategy']

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Validation Comparison Across Models and Strategies', fontsize=16)

    # 1. Accuracy comparison
    x_pos = np.arange(len(df))
    axes[0, 0].bar(x_pos, df['accuracy_mean'], yerr=df['accuracy_std'],
                   capsize=5, alpha=0.8)
    axes[0, 0].set_xlabel('Model + CV Strategy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(df['model_cv'], rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. F1-Score comparison
    axes[0, 1].bar(x_pos, df['f1_mean'], yerr=df['f1_std'],
                   capsize=5, alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Model + CV Strategy')
    axes[0, 1].set_ylabel('F1-Score (Macro)')
    axes[0, 1].set_title('F1-Score Comparison')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(df['model_cv'], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Overfitting analysis
    scatter = axes[1, 0].scatter(df['accuracy_mean'], df['overfitting_gap'],
                                s=100, alpha=0.7, c=df['cv_time'], cmap='viridis')
    axes[1, 0].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Mild overfitting threshold')
    axes[1, 0].axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='Significant overfitting threshold')
    axes[1, 0].set_xlabel('Test Accuracy')
    axes[1, 0].set_ylabel('Overfitting Gap (Train - Test)')
    axes[1, 0].set_title('Overfitting vs Accuracy (Color = CV Time)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='CV Time (seconds)')

    # 4. CV Time comparison
    axes[1, 1].bar(x_pos, df['cv_time'], alpha=0.8, color='green')
    axes[1, 1].set_xlabel('Model + CV Strategy')
    axes[1, 1].set_ylabel('CV Time (seconds)')
    axes[1, 1].set_title('Cross-Validation Time Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(df['model_cv'], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save comparison plot
    comparison_path = output_dir / "cv_comparison_all_models.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    return df, comparison_path

def main():
    parser = argparse.ArgumentParser(description="Comprehensive K-fold Cross-Validation with MLflow")
    parser.add_argument("--models", nargs='+',
                       help="Specific models to cross-validate (default: all)")
    parser.add_argument("--cv-strategies", nargs='+',
                       choices=['KFold', 'StratifiedKFold', 'ShuffleSplit', 'LeaveOneOut'],
                       default=['KFold', 'StratifiedKFold'],
                       help="CV strategies to use (default: KFold, StratifiedKFold)")
    parser.add_argument("--n-splits", type=int, default=5,
                       help="Number of CV splits (default: 5)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Base output directory")

    args = parser.parse_args()

    # Initialize output manager
    output_manager = OutputManager(args.output_dir)
    output_manager.logger.info("Starting Cross-Validation Analysis with MLflow")

    print("🎯 MLflow Cross-Validation Analysis - Iris Classification")
    print("=" * 65)

    # Set MLflow experiment
    mlflow.set_experiment("iris-cross-validation")

    # Prepare data
    print("📊 Preparing data...")
    X, y, class_names = prepare_data()

    print(f"Samples: {len(X)}")
    print(f"Features: {list(X.columns)}")
    print(f"Classes: {class_names}")

    # Get models and CV strategies
    models = get_model_configs()
    cv_strategies = get_cv_strategies(n_splits=args.n_splits, random_state=args.random_state)

    # Filter models if specified
    if args.models:
        available_models = list(models.keys())
        requested_models = [m for m in args.models if m in available_models]
        if not requested_models:
            print(f"❌ No valid models found. Available: {available_models}")
            return
        models = {k: v for k, v in models.items() if k in requested_models}
        print(f"🎯 Cross-validating specific models: {list(models.keys())}")
    else:
        print(f"🎯 Cross-validating all models: {list(models.keys())}")

    # Filter CV strategies
    if args.cv_strategies:
        cv_strategies = {k: v for k, v in cv_strategies.items() if k in args.cv_strategies}

    print(f"📋 Using CV strategies: {list(cv_strategies.keys())}")

    # Define scoring metrics
    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    # Perform cross-validation
    all_results = []
    total_experiments = len(models) * len(cv_strategies)
    current_exp = 0

    for model_name, model in models.items():
        for cv_name, cv_strategy in cv_strategies.items():
            current_exp += 1
            print(f"\n📈 Progress: {current_exp}/{total_experiments}")

            try:
                result = perform_cross_validation(
                    model_name=model_name,
                    model=model,
                    X=X,
                    y=y,
                    cv_strategy_name=cv_name,
                    cv_strategy=cv_strategy,
                    scoring_metrics=scoring_metrics,
                    class_names=class_names,
                    output_manager=output_manager
                )
                all_results.append(result)

            except Exception as e:
                print(f"❌ Error in {model_name} with {cv_name}: {str(e)}")

    # Create comparison analysis
    if all_results:
        print(f"\n📊 Creating comparison analysis...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = Path(f"outputs/results/cross_validation/comparison_{timestamp}")
        comparison_dir.mkdir(parents=True, exist_ok=True)

        comparison_df, comparison_viz_path = create_comparison_visualization(all_results, comparison_dir)

        # Save comparison results
        comparison_results_path = comparison_dir / "cv_comparison_results.csv"
        comparison_df.to_csv(comparison_results_path, index=False)

        # Log summary to MLflow
        with mlflow.start_run(run_name="Cross_Validation_Summary"):
            mlflow.log_param("total_experiments", len(all_results))
            mlflow.log_param("models_tested", list(models.keys()))
            mlflow.log_param("cv_strategies", list(cv_strategies.keys()))
            mlflow.log_param("n_splits", args.n_splits)

            # Find best model/strategy combination
            best_result = comparison_df.loc[comparison_df['accuracy_mean'].idxmax()]
            mlflow.log_param("best_model", best_result['model'])
            mlflow.log_param("best_cv_strategy", best_result['cv_strategy'])
            mlflow.log_metric("best_accuracy_mean", best_result['accuracy_mean'])
            mlflow.log_metric("best_accuracy_std", best_result['accuracy_std'])
            mlflow.log_metric("best_overfitting_gap", best_result['overfitting_gap'])

            # Log artifacts
            mlflow.log_artifact(str(comparison_viz_path))
            mlflow.log_artifact(str(comparison_results_path))

        # Print final summary
        print(f"\n🏆 CROSS-VALIDATION SUMMARY")
        print("=" * 50)
        print(f"📊 Total experiments: {len(all_results)}")
        print(f"🥇 Best combination: {best_result['model']} + {best_result['cv_strategy']}")
        print(f"🎯 Best accuracy: {best_result['accuracy_mean']:.4f} ± {best_result['accuracy_std']:.4f}")
        print(f"📉 Overfitting gap: {best_result['overfitting_gap']:.4f}")

        print(f"\n📋 Detailed Results:")
        print(comparison_df[['model', 'cv_strategy', 'accuracy_mean', 'accuracy_std', 'overfitting_gap']].round(4))

        print(f"\n✅ Results saved to: {comparison_dir}")
        print(f"📊 View results in MLflow UI: http://localhost:5001")
        print(f"🔍 Experiment: iris-cross-validation")

if __name__ == "__main__":
    main()