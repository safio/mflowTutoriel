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

def train_and_evaluate_model(model_name, model_config, X_train, X_test, y_train, y_test, class_names):
    """Train a single model and log results to MLflow"""

    with mlflow.start_run(run_name=f"{model_name}_comparison"):
        print(f"\n🔄 Training {model_name}...")

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

        # Log parameters
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

        # Create and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save to structured output using temp file first
        cm_temp_path = f"temp_confusion_matrix_{model_name.lower()}.png"
        plt.savefig(cm_temp_path, dpi=300, bbox_inches='tight')

        # Move to organized location (this function is not defined in original, need to import from structured version)
        cm_final_path = f"outputs/visualizations/confusion_matrices/confusion_matrix_{model_name.lower()}.png"
        import os
        os.makedirs(os.path.dirname(cm_final_path), exist_ok=True)
        os.rename(cm_temp_path, cm_final_path)

        mlflow.log_artifact(cm_final_path)
        plt.close()

        # Log classification report to organized location
        report = classification_report(y_test, y_pred, target_names=class_names)
        report_path = f"outputs/reports/classification/classification_report_{model_name.lower()}.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=mlflow.models.infer_signature(X_train, y_pred),
            input_example=X_train.iloc[:5]
        )

        # Print results
        print(f"✅ {model_name} Results:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"   CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   Training Time: {training_time:.3f}s")

        # Return results for comparison
        return {
            'model_name': model_name,
            'run_id': mlflow.active_run().info.run_id,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'cv_accuracy': cv_mean,
            'training_time': training_time,
            'all_metrics': metrics
        }

def create_comparison_visualization(results):
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

    # Save to organized location
    comparison_path = 'outputs/visualizations/comparison_plots/model_comparison.png'
    import os
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    return df_results

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
    print("📊 Preparing data...")
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
            print(f"❌ No valid models found. Available: {available_models}")
            return
        models = {k: v for k, v in models.items() if k in requested_models}
        print(f"🎯 Comparing specific models: {list(models.keys())}")
    else:
        print(f"🎯 Comparing all models: {list(models.keys())}")

    # Train and evaluate all models
    results = []

    for model_name, model_config in models.items():
        try:
            result = train_and_evaluate_model(
                model_name, model_config, X_train, X_test, y_train, y_test, class_names
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")

    # Create comparison visualization
    if results:
        print(f"\n📈 Creating comparison visualization...")
        df_results = create_comparison_visualization(results)

        # Log comparison results
        with mlflow.start_run(run_name="Model_Comparison_Summary"):
            # Log the comparison plot from organized location
            comparison_path = 'outputs/visualizations/comparison_plots/model_comparison.png'
            mlflow.log_artifact(comparison_path)

            # Log summary statistics
            best_accuracy = df_results.loc[df_results['accuracy'].idxmax()]
            best_f1 = df_results.loc[df_results['f1_macro'].idxmax()]
            fastest = df_results.loc[df_results['training_time'].idxmin()]

            mlflow.log_param("best_accuracy_model", best_accuracy['model_name'])
            mlflow.log_param("best_f1_model", best_f1['model_name'])
            mlflow.log_param("fastest_model", fastest['model_name'])

            mlflow.log_metric("best_accuracy_score", best_accuracy['accuracy'])
            mlflow.log_metric("best_f1_score", best_f1['f1_macro'])
            mlflow.log_metric("fastest_training_time", fastest['training_time'])

            # Save detailed results to organized location
            results_path = 'outputs/results/model_comparison_results.csv'
            import os
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            df_results.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)

        # Print summary
        print(f"\n🏆 COMPARISON SUMMARY")
        print("=" * 30)
        print(f"🥇 Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
        print(f"🥇 Best F1-Score: {best_f1['model_name']} ({best_f1['f1_macro']:.4f})")
        print(f"⚡ Fastest Training: {fastest['model_name']} ({fastest['training_time']:.3f}s)")

        print(f"\n📊 Detailed Results:")
        print(df_results[['model_name', 'accuracy', 'f1_macro', 'cv_accuracy', 'training_time']].round(4))

        print(f"\n✅ Results logged to MLflow experiment 'iris-model-comparison'")
        print(f"📈 View comparison visualization: {comparison_path}")
        print(f"📊 Results CSV: {results_path}")
        print(f"🗂️ All outputs organized in: {args.output_dir}")

if __name__ == "__main__":
    main()