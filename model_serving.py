#!/usr/bin/env python3
"""
MLflow Model Serving

Serve ML models from MLflow Model Registry via REST API.
Supports both registered models and models from specific runs.

Usage:
    # Serve a registered model from Production stage
    python model_serving.py --model-name iris-classifier --stage Production

    # Serve a specific model version
    python model_serving.py --model-name iris-classifier --version 1

    # Serve a model from a run ID
    python model_serving.py --run-id <run_id>

    # Custom host and port
    python model_serving.py --model-name iris-classifier --host 0.0.0.0 --port 8000

API Endpoints:
    GET  /health          - Health check
    GET  /model/info      - Model information
    POST /predict         - Make predictions
    POST /predict/batch   - Batch predictions
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mlflow_tutorial.experiments.mlflow_registry_manager import MLflowRegistryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global model holder
MODEL_INFO = {
    'model': None,
    'model_name': None,
    'version': None,
    'stage': None,
    'run_id': None,
    'features': None
}


def load_model_from_registry(model_name: str, version: Optional[str] = None, stage: Optional[str] = None):
    """Load a model from the MLflow Model Registry"""
    try:
        if version:
            model_uri = f"models:/{model_name}/{version}"
            logger.info(f"Loading model '{model_name}' version {version}")
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
            logger.info(f"Loading model '{model_name}' from '{stage}' stage")
        else:
            model_uri = f"models:/{model_name}/Production"
            logger.info(f"Loading model '{model_name}' from Production (default)")
            stage = "Production"

        model = mlflow.pyfunc.load_model(model_uri)

        # Get model version info
        client = MlflowClient()
        if not version and stage:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                version = versions[0].version
                run_id = versions[0].run_id
        else:
            model_version = client.get_model_version(model_name, version)
            run_id = model_version.run_id
            stage = model_version.current_stage

        MODEL_INFO.update({
            'model': model,
            'model_name': model_name,
            'version': version,
            'stage': stage,
            'run_id': run_id,
            'features': model.metadata.get_input_schema().input_names() if hasattr(model.metadata, 'get_input_schema') else None
        })

        logger.info(f"✅ Model loaded successfully: {model_name} v{version} ({stage})")
        return model

    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise


def load_model_from_run(run_id: str, artifact_path: str = "model"):
    """Load a model from a specific MLflow run"""
    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info(f"Loading model from run {run_id[:8]}...")

        model = mlflow.pyfunc.load_model(model_uri)

        MODEL_INFO.update({
            'model': model,
            'model_name': None,
            'version': None,
            'stage': None,
            'run_id': run_id,
            'features': model.metadata.get_input_schema().input_names() if hasattr(model.metadata, 'get_input_schema') else None
        })

        logger.info(f"✅ Model loaded successfully from run {run_id[:8]}")
        return model

    except Exception as e:
        logger.error(f"❌ Error loading model from run: {str(e)}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if MODEL_INFO['model'] is None:
        return jsonify({
            'status': 'error',
            'message': 'No model loaded'
        }), 503

    return jsonify({
        'status': 'healthy',
        'message': 'Model serving is operational',
        'model_info': {
            'model_name': MODEL_INFO['model_name'],
            'version': MODEL_INFO['version'],
            'stage': MODEL_INFO['stage'],
            'run_id': MODEL_INFO['run_id'][:8] + '...' if MODEL_INFO['run_id'] else None
        }
    }), 200


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if MODEL_INFO['model'] is None:
        return jsonify({
            'error': 'No model loaded'
        }), 503

    return jsonify({
        'model_name': MODEL_INFO['model_name'],
        'version': MODEL_INFO['version'],
        'stage': MODEL_INFO['stage'],
        'run_id': MODEL_INFO['run_id'],
        'features': MODEL_INFO['features'],
        'model_type': str(type(MODEL_INFO['model']))
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a single prediction

    Expected JSON format:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    or
    {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
    """
    if MODEL_INFO['model'] is None:
        return jsonify({'error': 'No model loaded'}), 503

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Handle different input formats
        if 'features' in data:
            # List format
            if MODEL_INFO['features']:
                input_data = pd.DataFrame([data['features']], columns=MODEL_INFO['features'])
            else:
                input_data = pd.DataFrame([data['features']])
        else:
            # Dictionary format
            input_data = pd.DataFrame([data])

        # Make prediction
        prediction = MODEL_INFO['model'].predict(input_data)

        # Convert numpy types to Python types for JSON serialization
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        elif isinstance(prediction, (np.integer, np.floating)):
            prediction = prediction.item()

        response = {
            'prediction': prediction[0] if isinstance(prediction, list) and len(prediction) == 1 else prediction,
            'model_info': {
                'model_name': MODEL_INFO['model_name'],
                'version': MODEL_INFO['version'],
                'stage': MODEL_INFO['stage']
            }
        }

        logger.info(f"Prediction made: {response['prediction']}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make batch predictions

    Expected JSON format:
    {
        "instances": [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 2.9, 4.3, 1.3]
        ]
    }
    or
    {
        "instances": [
            {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, ...},
            {"sepal length (cm)": 6.2, "sepal width (cm)": 2.9, ...}
        ]
    }
    """
    if MODEL_INFO['model'] is None:
        return jsonify({'error': 'No model loaded'}), 503

    try:
        data = request.get_json()

        if not data or 'instances' not in data:
            return jsonify({'error': 'No instances provided'}), 400

        instances = data['instances']

        # Convert to DataFrame
        if isinstance(instances[0], list):
            if MODEL_INFO['features']:
                input_data = pd.DataFrame(instances, columns=MODEL_INFO['features'])
            else:
                input_data = pd.DataFrame(instances)
        else:
            input_data = pd.DataFrame(instances)

        # Make predictions
        predictions = MODEL_INFO['model'].predict(input_data)

        # Convert numpy types to Python types
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        response = {
            'predictions': predictions,
            'count': len(predictions),
            'model_info': {
                'model_name': MODEL_INFO['model_name'],
                'version': MODEL_INFO['version'],
                'stage': MODEL_INFO['stage']
            }
        }

        logger.info(f"Batch prediction made: {len(predictions)} instances")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


def print_usage_examples(host: str, port: int):
    """Print usage examples"""
    base_url = f"http://{host}:{port}"

    print("\n" + "="*80)
    print("MODEL SERVING STARTED")
    print("="*80)
    print(f"\nServer running at: {base_url}")
    print(f"\nModel Information:")
    print(f"  Name:    {MODEL_INFO['model_name']}")
    print(f"  Version: {MODEL_INFO['version']}")
    print(f"  Stage:   {MODEL_INFO['stage']}")
    print(f"  Run ID:  {MODEL_INFO['run_id'][:8]}..." if MODEL_INFO['run_id'] else "  Run ID:  N/A")

    print("\n" + "="*80)
    print("API ENDPOINTS")
    print("="*80)

    print(f"\n1. Health Check:")
    print(f"   curl {base_url}/health")

    print(f"\n2. Model Info:")
    print(f"   curl {base_url}/model/info")

    print(f"\n3. Single Prediction:")
    print(f"   curl -X POST {base_url}/predict \\")
    print(f"     -H 'Content-Type: application/json' \\")
    print(f"     -d '{{\n         \"features\": [5.1, 3.5, 1.4, 0.2]\n       }}'")

    print(f"\n4. Batch Prediction:")
    print(f"   curl -X POST {base_url}/predict/batch \\")
    print(f"     -H 'Content-Type: application/json' \\")
    print(f"     -d '{{\n         \"instances\": [\n           [5.1, 3.5, 1.4, 0.2],\n           [6.2, 2.9, 4.3, 1.3]\n         ]\n       }}'")

    print("\n" + "="*80)
    print("\nPress Ctrl+C to stop the server\n")


def main():
    parser = argparse.ArgumentParser(
        description='Serve ML models from MLflow Model Registry',
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

    # Server options
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5002, help='Port number (default: 5002)')
    parser.add_argument('--artifact-path', type=str, default='model', help='Artifact path (for run-id)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    try:
        # Load the model
        if args.model_name:
            load_model_from_registry(
                model_name=args.model_name,
                version=args.version,
                stage=args.stage
            )
        elif args.run_id:
            load_model_from_run(
                run_id=args.run_id,
                artifact_path=args.artifact_path
            )

        # Print usage examples
        print_usage_examples(args.host, args.port)

        # Start the server
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
