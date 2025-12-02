#!/usr/bin/env python3
"""
Batch Prediction Script

Load data from CSV files and make batch predictions using models from MLflow Model Registry.
Supports saving predictions to CSV and logging results to MLflow.

Usage:
    # Predict using Production model
    python batch_prediction.py --model-name iris-classifier --input data.csv --output predictions.csv

    # Predict using specific version
    python batch_prediction.py --model-name iris-classifier --version 2 --input data.csv

    # Predict and log to MLflow
    python batch_prediction.py --model-name iris-classifier --input data.csv --log-to-mlflow

    # Use model from run ID
    python batch_prediction.py --run-id <run_id> --input data.csv --output predictions.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.output_manager import OutputManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchPredictor:
    """Handle batch predictions from MLflow models"""

    def __init__(self, output_manager: Optional[OutputManager] = None):
        self.client = MlflowClient()
        self.output_manager = output_manager or OutputManager()
        self.model = None
        self.model_info = {}

    def load_model_from_registry(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ):
        """Load model from MLflow Model Registry"""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
                logger.info(f"Loading model '{model_name}' version {version}")
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
                logger.info(f"Loading model '{model_name}' from '{stage}' stage")
            else:
                model_uri = f"models:/{model_name}/Production"
                logger.info(f"Loading model '{model_name}' from Production")
                stage = "Production"

            self.model = mlflow.pyfunc.load_model(model_uri)

            # Get model version info
            if not version and stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if versions:
                    version = versions[0].version
                    run_id = versions[0].run_id
            else:
                model_version = self.client.get_model_version(model_name, version)
                run_id = model_version.run_id
                stage = model_version.current_stage if not stage else stage

            self.model_info = {
                'model_name': model_name,
                'version': version,
                'stage': stage,
                'run_id': run_id
            }

            logger.info(f"✅ Model loaded: {model_name} v{version} ({stage})")

        except Exception as e:
            logger.error(f"❌ Error loading model from registry: {str(e)}")
            raise

    def load_model_from_run(self, run_id: str, artifact_path: str = "model"):
        """Load model from a specific run"""
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            logger.info(f"Loading model from run {run_id[:8]}...")

            self.model = mlflow.pyfunc.load_model(model_uri)

            self.model_info = {
                'model_name': None,
                'version': None,
                'stage': None,
                'run_id': run_id
            }

            logger.info(f"✅ Model loaded from run {run_id[:8]}")

        except Exception as e:
            logger.error(f"❌ Error loading model from run: {str(e)}")
            raise

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"✅ Loaded {len(data)} rows, {len(data.columns)} columns")
            return data
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on the data"""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model_from_registry() or load_model_from_run() first.")

        try:
            logger.info(f"Making predictions on {len(data)} instances...")

            # Make predictions
            predictions = self.model.predict(data)

            # Create results DataFrame
            results = data.copy()
            results['prediction'] = predictions

            # Add prediction timestamp
            results['prediction_timestamp'] = datetime.now().isoformat()

            logger.info(f"✅ Predictions completed")
            return results

        except Exception as e:
            logger.error(f"❌ Error making predictions: {str(e)}")
            raise

    def save_predictions(self, predictions: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Save predictions to CSV"""
        try:
            if output_path is None:
                # Use output manager to create timestamped file
                output_path = self.output_manager.get_results_path(
                    'testing',
                    f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            predictions.to_csv(output_path, index=False)
            logger.info(f"✅ Predictions saved to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ Error saving predictions: {str(e)}")
            raise

    def log_to_mlflow(self, predictions: pd.DataFrame, input_file: str):
        """Log batch prediction results to MLflow"""
        try:
            experiment_name = "batch-predictions"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("model_name", self.model_info.get('model_name', 'N/A'))
                mlflow.log_param("model_version", self.model_info.get('version', 'N/A'))
                mlflow.log_param("model_stage", self.model_info.get('stage', 'N/A'))
                mlflow.log_param("source_run_id", self.model_info.get('run_id', 'N/A'))
                mlflow.log_param("input_file", input_file)
                mlflow.log_param("num_predictions", len(predictions))

                # Log metrics
                mlflow.log_metric("batch_size", len(predictions))

                # Get prediction distribution
                pred_counts = predictions['prediction'].value_counts().to_dict()
                for pred_value, count in pred_counts.items():
                    mlflow.log_metric(f"prediction_class_{pred_value}_count", count)
                    mlflow.log_metric(f"prediction_class_{pred_value}_pct", (count / len(predictions)) * 100)

                # Save predictions as artifact
                temp_file = Path("temp_predictions.csv")
                predictions.to_csv(temp_file, index=False)
                mlflow.log_artifact(str(temp_file), "predictions")
                temp_file.unlink()

                run_id = mlflow.active_run().info.run_id
                logger.info(f"✅ Logged to MLflow (Run ID: {run_id[:8]}...)")

        except Exception as e:
            logger.error(f"❌ Error logging to MLflow: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Make batch predictions using MLflow models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model loading options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model-name', type=str, help='Name of the registered model')
    model_group.add_argument('--run-id', type=str, help='MLflow run ID')

    # Model version/stage
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument('--version', type=str, help='Model version number')
    version_group.add_argument('--stage', type=str,
                               choices=['Staging', 'Production', 'Archived'],
                               help='Model stage (default: Production)')

    # Data options
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output CSV file path (auto-generated if not provided)')

    # Other options
    parser.add_argument('--artifact-path', type=str, default='model', help='Artifact path (for run-id)')
    parser.add_argument('--log-to-mlflow', action='store_true', help='Log results to MLflow')
    parser.add_argument('--show-sample', type=int, help='Show N sample predictions')

    args = parser.parse_args()

    try:
        # Initialize predictor
        predictor = BatchPredictor()

        # Load model
        if args.model_name:
            predictor.load_model_from_registry(
                model_name=args.model_name,
                version=args.version,
                stage=args.stage
            )
        else:
            predictor.load_model_from_run(
                run_id=args.run_id,
                artifact_path=args.artifact_path
            )

        # Load data
        data = predictor.load_data(args.input)

        # Make predictions
        predictions = predictor.predict(data)

        # Save predictions
        output_path = predictor.save_predictions(predictions, args.output)

        # Show sample predictions if requested
        if args.show_sample:
            print("\n" + "="*80)
            print(f"SAMPLE PREDICTIONS (first {args.show_sample} rows)")
            print("="*80)
            print(predictions.head(args.show_sample).to_string(index=False))
            print("="*80)

        # Log to MLflow if requested
        if args.log_to_mlflow:
            predictor.log_to_mlflow(predictions, args.input)

        # Print summary
        print("\n" + "="*80)
        print("BATCH PREDICTION SUMMARY")
        print("="*80)
        print(f"Model:        {predictor.model_info.get('model_name', 'N/A')}")
        print(f"Version:      {predictor.model_info.get('version', 'N/A')}")
        print(f"Stage:        {predictor.model_info.get('stage', 'N/A')}")
        print(f"Input file:   {args.input}")
        print(f"Output file:  {output_path}")
        print(f"Total rows:   {len(predictions)}")
        print("\nPrediction Distribution:")
        print(predictions['prediction'].value_counts().to_string())
        print("="*80)

        logger.info("✅ Batch prediction completed successfully")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
