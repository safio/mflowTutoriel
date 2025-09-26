import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import argparse

def create_sample_data():
    """Create sample data points for testing"""
    # Create some sample iris-like data points
    sample_data = pd.DataFrame({
        'sepal length (cm)': [5.1, 6.2, 4.9, 7.0, 5.8],
        'sepal width (cm)': [3.5, 2.9, 3.0, 3.2, 2.7],
        'petal length (cm)': [1.4, 4.3, 1.4, 4.7, 4.1],
        'petal width (cm)': [0.2, 1.3, 0.2, 1.4, 1.0]
    })
    return sample_data

def load_model_from_mlflow(run_id=None, model_name="model"):
    """Load model from MLflow"""
    if run_id:
        # Load model from specific run
        model_uri = f"runs:/{run_id}/{model_name}"
        print(f"Loading model from run: {run_id}")
    else:
        # Load latest model from iris-classification experiment
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("iris-classification")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs:
                latest_run_id = runs[0].info.run_id
                model_uri = f"runs:/{latest_run_id}/{model_name}"
                print(f"Loading latest model from run: {latest_run_id}")
            else:
                raise ValueError("No runs found in iris-classification experiment")
        else:
            raise ValueError("iris-classification experiment not found")

    model = mlflow.sklearn.load_model(model_uri)
    return model, model_uri

def test_model_with_data(model, test_data, show_details=True):
    """Test model with provided data"""
    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data)

    # Get class names
    iris = load_iris()
    class_names = iris.target_names

    if show_details:
        print(f"\nTest Results:")
        print("-" * 50)
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"Sample {i+1}:")
            print(f"  Features: {test_data.iloc[i].to_dict()}")
            print(f"  Predicted class: {class_names[pred]} (class {pred})")
            print(f"  Probabilities: {dict(zip(class_names, prob.round(3)))}")
            print()

    return predictions, probabilities

def main():
    parser = argparse.ArgumentParser(description="Test trained iris classification model")
    parser.add_argument("--run-id", type=str, help="Specific MLflow run ID to load model from")
    parser.add_argument("--data-file", type=str, help="CSV file with test data (optional)")
    parser.add_argument("--use-sample", action="store_true", help="Use built-in sample data")
    parser.add_argument("--log-to-mlflow", action="store_true", help="Log test results to MLflow UI")
    args = parser.parse_args()

    try:
        # Load the trained model
        model, model_uri = load_model_from_mlflow(args.run_id)
        print(f"Model loaded from: {model_uri}")

        # Prepare test data
        if args.data_file:
            # Load data from CSV file
            print(f"\nLoading test data from: {args.data_file}")
            test_data = pd.read_csv(args.data_file)
        elif args.use_sample:
            # Use sample data
            print("\nUsing built-in sample data")
            test_data = create_sample_data()
        else:
            # Use some real iris data for demonstration
            iris = load_iris()
            test_data = pd.DataFrame(iris.data[:5], columns=iris.feature_names)
            print("\nUsing first 5 samples from original iris dataset")

        print(f"Test data shape: {test_data.shape}")
        print(f"Test data:\n{test_data}")

        # Test the model
        predictions, probabilities = test_model_with_data(model, test_data)

        # Log to MLflow if requested
        if args.log_to_mlflow:
            mlflow.set_experiment("iris-classification")

            with mlflow.start_run(run_name="Model_Test"):
                # Log test parameters
                mlflow.log_param("original_model_uri", model_uri)
                mlflow.log_param("test_data_source",
                    args.data_file if args.data_file else
                    "sample_data" if args.use_sample else "iris_original")
                mlflow.log_param("num_test_samples", len(test_data))

                # Calculate and log metrics
                iris = load_iris()
                class_counts = pd.Series(predictions).value_counts().to_dict()
                for class_idx, count in class_counts.items():
                    class_name = iris.target_names[class_idx]
                    mlflow.log_metric(f"predicted_{class_name}_count", count)

                # Log average confidence for each class
                avg_confidence = np.mean(np.max(probabilities, axis=1))
                mlflow.log_metric("average_prediction_confidence", avg_confidence)

                # Save test data and predictions as artifacts
                test_results = test_data.copy()
                test_results['predicted_class'] = [iris.target_names[p] for p in predictions]
                test_results['prediction_confidence'] = np.max(probabilities, axis=1)

                # Save as CSV artifact
                results_file = "test_results.csv"
                test_results.to_csv(results_file, index=False)
                mlflow.log_artifact(results_file)

                # Log individual prediction probabilities
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    for j, class_prob in enumerate(prob):
                        mlflow.log_metric(f"sample_{i+1}_{iris.target_names[j]}_prob", class_prob)

                print(f"\n✅ Test results logged to MLflow experiment 'iris-classification-tests'")
                print(f"Run ID: {mlflow.active_run().info.run_id}")

        # Summary
        iris = load_iris()
        print(f"\nSummary:")
        print(f"- Tested {len(test_data)} samples")
        print(f"- Predictions: {[iris.target_names[p] for p in predictions]}")
        print(f"- Average confidence: {np.mean(np.max(probabilities, axis=1)):.3f}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTo test your model:")
        print("1. First train a model: python train.py")
        print("2. Then test it: python test_model.py --use-sample")

if __name__ == "__main__":
    main()