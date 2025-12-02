#!/usr/bin/env python3
"""
Real-time Prediction Demo

Interactive demonstration of real-time predictions using models from MLflow Model Registry.
Shows how to load models and make instant predictions.

Usage:
    # Interactive mode with Production model
    python realtime_prediction.py --model-name iris-classifier

    # Use specific version
    python realtime_prediction.py --model-name iris-classifier --version 2

    # Single prediction from command line
    python realtime_prediction.py --model-name iris-classifier --predict 5.1,3.5,1.4,0.2

    # Test with sample data
    python realtime_prediction.py --model-name iris-classifier --demo
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealtimePredictor:
    """Handle real-time predictions from MLflow models"""

    def __init__(self):
        self.client = MlflowClient()
        self.model = None
        self.model_info = {}
        self.feature_names = None
        self.target_names = None

    def load_model(
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

            # Try to get feature names from model metadata
            try:
                if hasattr(self.model.metadata, 'get_input_schema'):
                    self.feature_names = self.model.metadata.get_input_schema().input_names()
            except:
                pass

            # For iris dataset, set standard names
            if not self.feature_names:
                iris = load_iris()
                self.feature_names = iris.feature_names
                self.target_names = iris.target_names

            logger.info(f"✅ Model loaded: {model_name} v{version} ({stage})")

        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise

    def predict_single(self, features: List[float]) -> dict:
        """Make a single prediction"""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        try:
            # Create DataFrame with feature names
            if self.feature_names and len(features) == len(self.feature_names):
                input_data = pd.DataFrame([features], columns=self.feature_names)
            else:
                input_data = pd.DataFrame([features])

            # Make prediction
            prediction = self.model.predict(input_data)

            # Extract prediction value
            if isinstance(prediction, np.ndarray):
                pred_value = int(prediction[0])
            else:
                pred_value = int(prediction)

            # Get class name if available
            class_name = self.target_names[pred_value] if self.target_names else None

            return {
                'prediction': pred_value,
                'class_name': class_name,
                'features': features,
                'feature_names': self.feature_names
            }

        except Exception as e:
            logger.error(f"❌ Error making prediction: {str(e)}")
            raise

    def interactive_mode(self):
        """Run interactive prediction mode"""
        print("\n" + "="*80)
        print("REAL-TIME PREDICTION - INTERACTIVE MODE")
        print("="*80)
        print(f"\nModel: {self.model_info['model_name']} v{self.model_info['version']} ({self.model_info['stage']})")

        if self.feature_names:
            print(f"\nFeatures required: {len(self.feature_names)}")
            for i, name in enumerate(self.feature_names, 1):
                print(f"  {i}. {name}")

        if self.target_names:
            print(f"\nPossible classes:")
            for i, name in enumerate(self.target_names):
                print(f"  {i}. {name}")

        print("\n" + "="*80)
        print("Enter feature values separated by commas (or 'quit' to exit)")
        print("Example: 5.1,3.5,1.4,0.2")
        print("="*80)

        while True:
            try:
                user_input = input("\n> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                # Parse input
                try:
                    features = [float(x.strip()) for x in user_input.split(',')]
                except ValueError:
                    print("❌ Invalid input. Please enter numeric values separated by commas.")
                    continue

                # Validate number of features
                if self.feature_names and len(features) != len(self.feature_names):
                    print(f"❌ Expected {len(self.feature_names)} features, got {len(features)}")
                    continue

                # Make prediction
                result = self.predict_single(features)

                # Display result
                print("\n" + "-"*40)
                print(f"Prediction:  {result['prediction']}")
                if result['class_name']:
                    print(f"Class:       {result['class_name']}")
                print("-"*40)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def demo_mode(self):
        """Run demo with sample data"""
        print("\n" + "="*80)
        print("REAL-TIME PREDICTION - DEMO MODE")
        print("="*80)
        print(f"\nModel: {self.model_info['model_name']} v{self.model_info['version']} ({self.model_info['stage']})")

        # Load sample data
        iris = load_iris()
        X_sample = iris.data[:10]  # First 10 samples
        y_actual = iris.target[:10]

        print(f"\nMaking predictions on {len(X_sample)} sample instances...")
        print("\n" + "="*80)

        correct = 0
        for i, (features, actual) in enumerate(zip(X_sample, y_actual), 1):
            result = self.predict_single(features.tolist())

            # Show result
            print(f"\nSample {i}:")
            print(f"  Features:   {[f'{f:.1f}' for f in features]}")
            print(f"  Predicted:  {result['prediction']} ({result['class_name']})")
            print(f"  Actual:     {actual} ({iris.target_names[actual]})")
            print(f"  ✓ Correct" if result['prediction'] == actual else f"  ✗ Incorrect")

            if result['prediction'] == actual:
                correct += 1

        # Summary
        accuracy = (correct / len(X_sample)) * 100
        print("\n" + "="*80)
        print(f"DEMO SUMMARY")
        print("="*80)
        print(f"Total samples:  {len(X_sample)}")
        print(f"Correct:        {correct}")
        print(f"Accuracy:       {accuracy:.1f}%")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Real-time prediction demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model options
    parser.add_argument('--model-name', type=str, required=True, help='Name of the registered model')

    # Model version/stage
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument('--version', type=str, help='Model version number')
    version_group.add_argument('--stage', type=str,
                               choices=['Staging', 'Production', 'Archived'],
                               help='Model stage (default: Production)')

    # Prediction options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--predict', type=str, help='Make single prediction (comma-separated values)')
    mode_group.add_argument('--demo', action='store_true', help='Run demo mode with sample data')

    args = parser.parse_args()

    try:
        # Initialize predictor
        predictor = RealtimePredictor()

        # Load model
        predictor.load_model(
            model_name=args.model_name,
            version=args.version,
            stage=args.stage
        )

        # Run appropriate mode
        if args.predict:
            # Single prediction from command line
            features = [float(x.strip()) for x in args.predict.split(',')]
            result = predictor.predict_single(features)

            print("\n" + "="*80)
            print("PREDICTION RESULT")
            print("="*80)
            print(f"Features:    {features}")
            print(f"Prediction:  {result['prediction']}")
            if result['class_name']:
                print(f"Class:       {result['class_name']}")
            print("="*80)

        elif args.demo:
            # Demo mode
            predictor.demo_mode()

        else:
            # Interactive mode (default)
            predictor.interactive_mode()

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
