import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Set MLflow experiment (required even for projects)
    mlflow.set_experiment("iris-classification")

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

    print(f"\nData split:")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Test samples: {len(X_test)}")
    print(f"- Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"- Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("total_samples", len(X))
    mlflow.log_param("training_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model with signature and input example
    signature = mlflow.models.infer_signature(X_train, y_pred)
    input_example = X_train.iloc[:5]
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=input_example
    )

    print(f"Accuracy: {accuracy:.4f}")
    if mlflow.active_run():
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()