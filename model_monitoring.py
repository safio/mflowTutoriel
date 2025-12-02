#!/usr/bin/env python3
"""
Model Performance Monitoring

Monitor model performance over time by tracking predictions and metrics.
Helps detect model degradation, data drift, and when retraining is needed.

Usage:
    # Monitor a specific model version
    python model_monitoring.py --model-name iris-classifier --stage Production

    # Generate performance report
    python model_monitoring.py --model-name iris-classifier --report

    # Track prediction history
    python model_monitoring.py --model-name iris-classifier --track-predictions --input data.csv --labels labels.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.output_manager import OutputManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance over time"""

    def __init__(self):
        self.client = MlflowClient()
        self.output_manager = OutputManager()
        self.monitoring_log = self.output_manager.get_results_path('monitoring', 'performance_log.csv')

    def track_prediction_batch(
        self,
        model_name: str,
        version: str,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track a batch of predictions for monitoring

        Args:
            model_name: Name of the model
            version: Model version
            predictions: Model predictions
            actuals: Actual labels (if available)
            metadata: Additional metadata (e.g., data source, timestamp)
        """
        timestamp = datetime.now()

        # Calculate metrics if actuals are provided
        metrics = {}
        if actuals is not None:
            metrics = {
                'accuracy': accuracy_score(actuals, predictions),
                'precision': precision_score(actuals, predictions, average='weighted', zero_division=0),
                'recall': recall_score(actuals, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(actuals, predictions, average='weighted', zero_division=0)
            }

        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'model_name': model_name,
            'version': version,
            'batch_size': len(predictions),
            'has_labels': actuals is not None,
            **metrics,
            **(metadata or {})
        }

        # Append to monitoring log
        log_df = pd.DataFrame([log_entry])

        if Path(self.monitoring_log).exists():
            existing_df = pd.read_csv(self.monitoring_log)
            log_df = pd.concat([existing_df, log_df], ignore_index=True)

        log_df.to_csv(self.monitoring_log, index=False)
        logger.info(f"✅ Logged prediction batch: {len(predictions)} predictions")

        return log_entry

    def get_performance_history(
        self,
        model_name: str,
        version: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get performance history for a model

        Args:
            model_name: Name of the model
            version: Specific version (None for all versions)
            days: Number of days to look back

        Returns:
            DataFrame with performance history
        """
        if not Path(self.monitoring_log).exists():
            logger.warning("No monitoring log found")
            return pd.DataFrame()

        df = pd.read_csv(self.monitoring_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by model name
        df = df[df['model_name'] == model_name]

        # Filter by version if specified
        if version:
            df = df[df['version'] == version]

        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff_date]

        return df.sort_values('timestamp')

    def detect_performance_degradation(
        self,
        model_name: str,
        version: Optional[str] = None,
        metric: str = 'accuracy',
        threshold: float = 0.05,
        window_size: int = 5
    ) -> Dict[str, Any]:
        """
        Detect if model performance has degraded

        Args:
            model_name: Name of the model
            version: Model version
            metric: Metric to monitor
            threshold: Degradation threshold (e.g., 0.05 = 5% drop)
            window_size: Number of recent batches to compare

        Returns:
            Dictionary with degradation analysis
        """
        history = self.get_performance_history(model_name, version)

        if len(history) < window_size * 2:
            return {
                'degraded': False,
                'reason': 'Insufficient data for analysis',
                'current_performance': None,
                'baseline_performance': None
            }

        # Only consider entries with labels
        history = history[history['has_labels'] == True]

        if len(history) < window_size * 2:
            return {
                'degraded': False,
                'reason': 'Insufficient labeled data for analysis',
                'current_performance': None,
                'baseline_performance': None
            }

        # Get recent and baseline performance
        recent_perf = history[metric].tail(window_size).mean()
        baseline_perf = history[metric].head(window_size).mean()

        # Check for degradation
        degradation = baseline_perf - recent_perf
        degradation_pct = (degradation / baseline_perf) * 100 if baseline_perf > 0 else 0

        is_degraded = degradation > threshold

        return {
            'degraded': is_degraded,
            'metric': metric,
            'current_performance': recent_perf,
            'baseline_performance': baseline_perf,
            'degradation': degradation,
            'degradation_pct': degradation_pct,
            'threshold': threshold,
            'recommendation': 'Consider retraining the model' if is_degraded else 'Performance is stable'
        }

    def plot_performance_over_time(
        self,
        model_name: str,
        version: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        days: int = 30,
        output_path: Optional[str] = None
    ) -> str:
        """
        Plot model performance over time

        Args:
            model_name: Name of the model
            version: Model version
            metrics: List of metrics to plot
            days: Number of days to include
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        history = self.get_performance_history(model_name, version, days)

        if history.empty:
            logger.warning("No performance history found")
            return None

        # Filter to entries with labels
        history = history[history['has_labels'] == True]

        if history.empty:
            logger.warning("No labeled performance history found")
            return None

        # Default metrics
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        # Filter to available metrics
        available_metrics = [m for m in metrics if m in history.columns and not history[m].isna().all()]

        if not available_metrics:
            logger.warning("No metrics available for plotting")
            return None

        # Create plot
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 4 * len(available_metrics)))

        if len(available_metrics) == 1:
            axes = [axes]

        fig.suptitle(f'Model Performance Over Time: {model_name}', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]

            # Plot metric over time
            ax.plot(history['timestamp'], history[metric], marker='o', linestyle='-', linewidth=2, markersize=6)

            # Add trend line
            z = np.polyfit(range(len(history)), history[metric], 1)
            p = np.poly1d(z)
            ax.plot(history['timestamp'], p(range(len(history))), linestyle='--', color='red', alpha=0.5, label='Trend')

            ax.set_xlabel('Date')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_manager.get_visualization_path(
                'comparison_plots',
                f'performance_monitoring_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Performance plot saved to {output_path}")
        plt.close()

        return output_path

    def generate_monitoring_report(
        self,
        model_name: str,
        version: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report

        Args:
            model_name: Name of the model
            version: Model version
            days: Number of days to include

        Returns:
            Dictionary with report data
        """
        history = self.get_performance_history(model_name, version, days)

        if history.empty:
            return {
                'model_name': model_name,
                'version': version,
                'status': 'No data available',
                'total_predictions': 0
            }

        # Filter to labeled data for metrics
        labeled_history = history[history['has_labels'] == True]

        # Calculate summary statistics
        summary = {
            'model_name': model_name,
            'version': version or 'All versions',
            'monitoring_period_days': days,
            'total_prediction_batches': len(history),
            'total_predictions': int(history['batch_size'].sum()),
            'labeled_batches': len(labeled_history),
            'labeled_predictions': int(labeled_history['batch_size'].sum()) if not labeled_history.empty else 0
        }

        # Add performance metrics if available
        if not labeled_history.empty:
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in labeled_history.columns:
                    summary[f'{metric}_mean'] = labeled_history[metric].mean()
                    summary[f'{metric}_std'] = labeled_history[metric].std()
                    summary[f'{metric}_min'] = labeled_history[metric].min()
                    summary[f'{metric}_max'] = labeled_history[metric].max()

        # Check for degradation
        degradation_analysis = self.detect_performance_degradation(model_name, version)
        summary['degradation_detected'] = degradation_analysis['degraded']
        summary['degradation_info'] = degradation_analysis

        return summary

    def print_report(self, report: Dict[str, Any]):
        """Print formatted monitoring report"""
        print("\n" + "="*80)
        print("MODEL MONITORING REPORT")
        print("="*80)

        print(f"\nModel:                {report['model_name']}")
        print(f"Version:              {report['version']}")
        print(f"Monitoring Period:    {report['monitoring_period_days']} days")

        print("\n" + "-"*80)
        print("PREDICTION SUMMARY")
        print("-"*80)
        print(f"Total Batches:        {report['total_prediction_batches']}")
        print(f"Total Predictions:    {report['total_predictions']}")
        print(f"Labeled Batches:      {report['labeled_batches']}")
        print(f"Labeled Predictions:  {report['labeled_predictions']}")

        if report['labeled_batches'] > 0:
            print("\n" + "-"*80)
            print("PERFORMANCE METRICS (Mean ± Std)")
            print("-"*80)
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in report:
                    print(f"{metric.replace('_', ' ').title():20} {report[mean_key]:.4f} ± {report[std_key]:.4f}")

        print("\n" + "-"*80)
        print("DEGRADATION ANALYSIS")
        print("-"*80)
        deg_info = report['degradation_info']
        print(f"Status:               {'⚠️  DEGRADATION DETECTED' if deg_info['degraded'] else '✅ STABLE'}")
        if deg_info['degraded']:
            print(f"Metric:               {deg_info['metric']}")
            print(f"Baseline:             {deg_info['baseline_performance']:.4f}")
            print(f"Current:              {deg_info['current_performance']:.4f}")
            print(f"Degradation:          {deg_info['degradation']:.4f} ({deg_info['degradation_pct']:.2f}%)")
            print(f"Recommendation:       {deg_info['recommendation']}")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Model performance monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--model-name', type=str, required=True, help='Name of the model')
    parser.add_argument('--version', type=str, help='Model version')
    parser.add_argument('--stage', type=str, choices=['Staging', 'Production', 'Archived'], help='Model stage')

    # Actions
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--report', action='store_true', help='Generate monitoring report')
    action_group.add_argument('--track-predictions', action='store_true', help='Track new predictions')
    action_group.add_argument('--plot', action='store_true', help='Plot performance over time')
    action_group.add_argument('--check-degradation', action='store_true', help='Check for performance degradation')

    # Tracking options
    parser.add_argument('--input', type=str, help='Input data CSV for predictions')
    parser.add_argument('--labels', type=str, help='True labels CSV')

    # Monitoring options
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--metrics', nargs='*', help='Metrics to monitor')
    parser.add_argument('--threshold', type=float, default=0.05, help='Degradation threshold (default: 0.05)')

    args = parser.parse_args()

    try:
        monitor = ModelMonitor()

        if args.track_predictions:
            # Track new predictions
            if not args.input:
                print("❌ Error: --input required for tracking predictions")
                sys.exit(1)

            # Load model and make predictions
            if args.stage:
                model_uri = f"models:/{args.model_name}/{args.stage}"
            elif args.version:
                model_uri = f"models:/{args.model_name}/{args.version}"
            else:
                model_uri = f"models:/{args.model_name}/Production"

            model = mlflow.pyfunc.load_model(model_uri)
            data = pd.read_csv(args.input)

            predictions = model.predict(data)

            # Load labels if provided
            actuals = None
            if args.labels:
                labels_df = pd.read_csv(args.labels)
                actuals = labels_df.iloc[:, 0].values

            # Track predictions
            version = args.version or args.stage or 'Production'
            monitor.track_prediction_batch(
                model_name=args.model_name,
                version=version,
                predictions=predictions,
                actuals=actuals,
                metadata={'data_source': args.input}
            )

            print(f"✅ Tracked {len(predictions)} predictions")

        elif args.report:
            # Generate report
            report = monitor.generate_monitoring_report(
                model_name=args.model_name,
                version=args.version,
                days=args.days
            )
            monitor.print_report(report)

            # Also create plot
            monitor.plot_performance_over_time(
                model_name=args.model_name,
                version=args.version,
                metrics=args.metrics,
                days=args.days
            )

        elif args.plot:
            # Plot performance
            plot_path = monitor.plot_performance_over_time(
                model_name=args.model_name,
                version=args.version,
                metrics=args.metrics,
                days=args.days
            )
            if plot_path:
                print(f"✅ Plot saved to {plot_path}")

        elif args.check_degradation:
            # Check for degradation
            result = monitor.detect_performance_degradation(
                model_name=args.model_name,
                version=args.version,
                threshold=args.threshold
            )

            print("\n" + "="*80)
            print("DEGRADATION CHECK")
            print("="*80)
            print(f"Model:        {args.model_name}")
            print(f"Status:       {'⚠️  DEGRADED' if result['degraded'] else '✅ STABLE'}")
            if result['current_performance'] is not None:
                print(f"Baseline:     {result['baseline_performance']:.4f}")
                print(f"Current:      {result['current_performance']:.4f}")
                print(f"Degradation:  {result.get('degradation', 0):.4f}")
                print(f"Threshold:    {result['threshold']:.4f}")
            print(f"Recommendation: {result['recommendation']}")
            print("="*80)

        else:
            # Default: show report
            report = monitor.generate_monitoring_report(
                model_name=args.model_name,
                version=args.version,
                days=args.days
            )
            monitor.print_report(report)

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
