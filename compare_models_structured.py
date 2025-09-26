import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from utils.output_manager import OutputManager

def prepare_data(test_size=0.2, random_state=42):
    """Load and prepare the iris dataset"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, iris.target_names

def get_model_configs():
    """Define all models to compare with their configurations"""
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': 100, 'max_depth': None}
        },
        'SVM_RBF': {
            'model': SVC(kernel='rbf', random_state=42, probability=True),
            'params': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
        },
        'SVM_Linear': {
            'model': SVC(kernel='linear', random_state=42, probability=True),
            'params': {'C': 1.0, 'kernel': 'linear'}
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {'C': 1.0, 'max_iter': 1000}
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': None, 'min_samples_split': 2}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': 5, 'weights': 'uniform'}
        },
        'NaiveBayes': {
            'model': GaussianNB(),
            'params': {'var_smoothing': 1e-9}
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
        }
    }
    return models

def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive metrics for model evaluation"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }

    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    class_names = ['setosa', 'versicolor', 'virginica']
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision_per_class[i]
        metrics[f'recall_{class_name}'] = recall_per_class[i]
        metrics[f'f1_{class_name}'] = f1_per_class[i]

    return metrics

def train_and_evaluate_model(model_name, model_config, X_train, X_test, y_train, y_test,
                           class_names, output_manager):
    """Train a single model and log results to MLflow and structured outputs"""

    with mlflow.start_run(run_name=f"{model_name}_comparison"):
        output_manager.logger.info(f"Training {model_name}...")

        # Record training start time
        start_time = time.time()

        # Train the model
        model = model_config['model']
        model.fit(X_train, y_train)

        # Record training time
        training_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_test)

        # Get prediction probabilities if available
        try:
            y_proba = model.predict_proba(X_test)
        except AttributeError:
            y_proba = None

        # Calculate metrics
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Prepare comprehensive results
        results = {
            'model_name': model_name,
            'run_id': mlflow.active_run().info.run_id,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time,
            'all_metrics': metrics,
            'model_params': model_config['params']
        }

        # Log to MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Log model-specific parameters
        for param_name, param_value in model_config['params'].items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("cv_accuracy_mean", cv_mean)
        mlflow.log_metric("cv_accuracy_std", cv_std)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Create and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save to structured output
        cm_temp_path = f"temp_confusion_matrix_{model_name.lower()}.png"
        plt.savefig(cm_temp_path, dpi=300, bbox_inches='tight')
        cm_final_path = output_manager.save_visualization(cm_temp_path, "confusion_matrices")

        # Log to MLflow
        mlflow.log_artifact(str(cm_final_path))
        plt.close()

        # Create and save classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        report_path = output_manager.get_output_path("reports/classification",
                                                   f"classification_report_{model_name.lower()}.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)
        mlflow.log_artifact(str(report_path))

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=mlflow.models.infer_signature(X_train, y_pred),
            input_example=X_train.iloc[:5]
        )

        # Save structured results
        output_manager.save_model_results(model_name, results, mlflow.active_run().info.run_id)

        # Log results
        output_manager.logger.info(f"{model_name} Results - Accuracy: {metrics['accuracy']:.4f}, "
                                 f"F1: {metrics['f1_macro']:.4f}, Time: {training_time:.3f}s")

        return results

def create_comparison_visualization(results, output_manager):
    """Create visualization comparing all models"""

    df_results = pd.DataFrame(results)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy comparison
    axes[0, 0].bar(df_results['model_name'], df_results['accuracy'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # F1-Score comparison
    axes[0, 1].bar(df_results['model_name'], df_results['f1_macro'])
    axes[0, 1].set_title('F1-Score (Macro) Comparison')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Cross-validation accuracy
    axes[1, 0].bar(df_results['model_name'], df_results['cv_accuracy'])
    axes[1, 0].set_title('Cross-Validation Accuracy')
    axes[1, 0].set_ylabel('CV Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Training time comparison
    axes[1, 1].bar(df_results['model_name'], df_results['training_time'])
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save to structured output
    temp_path = 'temp_model_comparison.png'
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    final_path = output_manager.save_visualization(temp_path, "comparison_plots")
    plt.close()

    return df_results, final_path

def main():
    parser = argparse.ArgumentParser(description="Compare different ML algorithms on iris dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--models", nargs='+', help="Specific models to compare (optional)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    args = parser.parse_args()

    # Initialize output manager
    output_manager = OutputManager(args.output_dir)

    output_manager.logger.info("Starting MLflow Model Comparison - Iris Classification")
    print("🌸 MLflow Model Comparison - Iris Classification")
    print("=" * 60)

    # Set MLflow experiment
    mlflow.set_experiment("iris-model-comparison")

    # Prepare data
    output_manager.logger.info("Preparing data...")
    X_train, X_test, y_train, y_test, class_names = prepare_data(
        test_size=args.test_size,
        random_state=args.random_state
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {class_names}")

    # Get model configurations
    models = get_model_configs()

    # Filter models if specific ones requested
    if args.models:
        available_models = list(models.keys())
        requested_models = [m for m in args.models if m in available_models]
        if not requested_models:
            output_manager.logger.error(f"No valid models found. Available: {available_models}")
            return
        models = {k: v for k, v in models.items() if k in requested_models}
        output_manager.logger.info(f"Comparing specific models: {list(models.keys())}")
    else:
        output_manager.logger.info(f"Comparing all models: {list(models.keys())}")

    # Train and evaluate all models
    results = []

    for model_name, model_config in models.items():
        try:
            result = train_and_evaluate_model(
                model_name, model_config, X_train, X_test, y_train, y_test,
                class_names, output_manager
            )
            results.append(result)
        except Exception as e:
            output_manager.logger.error(f"Error training {model_name}: {str(e)}")

    # Create comparison visualization and save results
    if results:
        output_manager.logger.info("Creating comparison visualization...")
        df_results, comparison_plot_path = create_comparison_visualization(results, output_manager)

        # Calculate summary statistics
        best_accuracy = df_results.loc[df_results['accuracy'].idxmax()]
        best_f1 = df_results.loc[df_results['f1_macro'].idxmax()]
        fastest = df_results.loc[df_results['training_time'].idxmin()]

        summary_stats = {
            'best_accuracy_model': best_accuracy['model_name'],
            'best_accuracy_score': best_accuracy['accuracy'],
            'best_f1_model': best_f1['model_name'],
            'best_f1_score': best_f1['f1_macro'],
            'fastest_model': fastest['model_name'],
            'fastest_time': fastest['training_time'],
            'total_models': len(results),
            'average_accuracy': df_results['accuracy'].mean(),
            'average_training_time': df_results['training_time'].mean()
        }

        # Save comparison results
        saved_files = output_manager.save_comparison_results(df_results, summary_stats)

        # Log comparison results to MLflow
        with mlflow.start_run(run_name="Model_Comparison_Summary"):
            mlflow.log_artifact(str(comparison_plot_path))
            mlflow.log_artifact(str(saved_files['results_csv']))
            mlflow.log_artifact(str(saved_files['summary']))

            # Log summary metrics
            for key, value in summary_stats.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(key, value)

        # Create comprehensive report
        report_summary = {
            'overview': f"Compared {len(results)} machine learning algorithms on iris classification",
            'key_results': {
                'Best Accuracy': f"{best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})",
                'Best F1-Score': f"{best_f1['model_name']} ({best_f1['f1_macro']:.4f})",
                'Fastest Training': f"{fastest['model_name']} ({fastest['training_time']:.3f}s)"
            },
            'recommendations': [
                f"Use {best_accuracy['model_name']} for highest accuracy",
                f"Use {fastest['model_name']} for fastest training",
                "Consider cross-validation scores for more robust evaluation"
            ],
            'files': [str(path) for path in saved_files.values()] + [str(comparison_plot_path)]
        }

        report_path = output_manager.create_summary_report("model_comparison", report_summary)

        # Print summary
        print(f"\n🏆 COMPARISON SUMMARY")
        print("=" * 30)
        print(f"🥇 Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
        print(f"🥇 Best F1-Score: {best_f1['model_name']} ({best_f1['f1_macro']:.4f})")
        print(f"⚡ Fastest Training: {fastest['model_name']} ({fastest['training_time']:.3f}s)")

        print(f"\n📊 Detailed Results:")
        print(df_results[['model_name', 'accuracy', 'f1_macro', 'cv_accuracy', 'training_time']].round(4))

        print(f"\n📁 ORGANIZED OUTPUTS:")
        print(f"📈 Comparison plot: {comparison_plot_path}")
        print(f"📊 Results CSV: {saved_files['results_csv']}")
        print(f"📋 Summary report: {report_path}")
        print(f"🗂️ All outputs organized in: {output_manager.base_dir}")

        output_manager.logger.info("Model comparison completed successfully")
        output_manager.logger.info(f"Results saved to {output_manager.base_dir}")

if __name__ == "__main__":
    main()