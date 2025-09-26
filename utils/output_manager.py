import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

class OutputManager:
    """Manages structured output directories and file organization"""

    def __init__(self, base_output_dir: str = "outputs"):
        self.base_dir = Path(base_output_dir)
        self.setup_directories()
        self.setup_logging()

    def setup_directories(self):
        """Create structured directory layout"""
        directories = [
            "logs",
            "results/model_comparison",
            "results/individual_models",
            "results/hyperparameter_tuning",
            "results/testing",
            "visualizations/confusion_matrices",
            "visualizations/comparison_plots",
            "visualizations/data_exploration",
            "models/trained",
            "models/artifacts",
            "data/processed",
            "data/test_samples",
            "reports/classification",
            "reports/summary",
            "configs"
        ]

        for dir_path in directories:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup structured logging configuration"""
        log_dir = self.base_dir / "logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )

        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Setup main logger
        self.logger = logging.getLogger('mlflow_tutorial')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler for detailed logs
        detailed_handler = logging.FileHandler(
            log_dir / f"detailed_{timestamp}.log"
        )
        detailed_handler.setLevel(logging.DEBUG)
        detailed_handler.setFormatter(detailed_formatter)

        # File handler for simple logs
        simple_handler = logging.FileHandler(
            log_dir / f"training_{timestamp}.log"
        )
        simple_handler.setLevel(logging.INFO)
        simple_handler.setFormatter(simple_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)

        # Add handlers
        self.logger.addHandler(detailed_handler)
        self.logger.addHandler(simple_handler)
        self.logger.addHandler(console_handler)

    def get_output_path(self, category: str, filename: str,
                       timestamp: bool = True) -> Path:
        """Get organized output path for a file"""
        if timestamp:
            base_name = Path(filename).stem
            extension = Path(filename).suffix
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp_str}{extension}"

        return self.base_dir / category / filename

    def save_model_results(self, model_name: str, results: Dict[str, Any],
                          run_id: str) -> Dict[str, Path]:
        """Save individual model results in organized structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.base_dir / "results" / "individual_models" / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save metrics as JSON
        metrics_file = model_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        saved_files['metrics'] = metrics_file

        # Save metadata
        metadata = {
            'model_name': model_name,
            'run_id': run_id,
            'timestamp': timestamp,
            'mlflow_experiment': 'iris-model-comparison'
        }
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_file

        self.logger.info(f"Saved {model_name} results to {model_dir}")
        return saved_files

    def save_comparison_results(self, results_df: pd.DataFrame,
                               summary_stats: Dict[str, Any]) -> Dict[str, Path]:
        """Save model comparison results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = self.base_dir / "results" / "model_comparison" / f"comparison_{timestamp}"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save results DataFrame
        results_file = comparison_dir / "comparison_results.csv"
        results_df.to_csv(results_file, index=False)
        saved_files['results_csv'] = results_file

        # Save summary statistics
        summary_file = comparison_dir / "summary_stats.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        saved_files['summary'] = summary_file

        # Save detailed analysis
        analysis = self._generate_detailed_analysis(results_df)
        analysis_file = comparison_dir / "detailed_analysis.txt"
        with open(analysis_file, 'w') as f:
            f.write(analysis)
        saved_files['analysis'] = analysis_file

        self.logger.info(f"Saved comparison results to {comparison_dir}")
        return saved_files

    def save_visualization(self, fig_path: str, category: str = "comparison_plots") -> Path:
        """Save visualization to appropriate directory"""
        filename = Path(fig_path).name
        target_path = self.get_output_path(f"visualizations/{category}", filename)

        # Move file to organized location
        if os.path.exists(fig_path):
            os.rename(fig_path, target_path)
            self.logger.info(f"Moved visualization to {target_path}")

        return target_path

    def save_test_results(self, test_data: pd.DataFrame, predictions: Dict[str, Any],
                         test_source: str) -> Dict[str, Path]:
        """Save model testing results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = self.base_dir / "results" / "testing" / f"test_{timestamp}"
        test_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save test data
        test_data_file = test_dir / "test_data.csv"
        test_data.to_csv(test_data_file, index=False)
        saved_files['test_data'] = test_data_file

        # Save predictions
        predictions_file = test_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        saved_files['predictions'] = predictions_file

        # Save test metadata
        metadata = {
            'test_source': test_source,
            'timestamp': timestamp,
            'num_samples': len(test_data)
        }
        metadata_file = test_dir / "test_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_file

        self.logger.info(f"Saved test results to {test_dir}")
        return saved_files

    def _generate_detailed_analysis(self, results_df: pd.DataFrame) -> str:
        """Generate detailed text analysis of comparison results"""
        analysis = []
        analysis.append("=" * 60)
        analysis.append("MODEL COMPARISON DETAILED ANALYSIS")
        analysis.append("=" * 60)
        analysis.append(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analysis.append("")

        # Overall statistics
        analysis.append("OVERALL STATISTICS:")
        analysis.append("-" * 20)
        analysis.append(f"Number of models compared: {len(results_df)}")
        analysis.append(f"Best accuracy: {results_df['accuracy'].max():.4f}")
        analysis.append(f"Worst accuracy: {results_df['accuracy'].min():.4f}")
        analysis.append(f"Average accuracy: {results_df['accuracy'].mean():.4f}")
        analysis.append(f"Accuracy std dev: {results_df['accuracy'].std():.4f}")
        analysis.append("")

        # Top performers
        analysis.append("TOP PERFORMERS:")
        analysis.append("-" * 15)
        top_3 = results_df.nlargest(3, 'accuracy')
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            analysis.append(f"{i}. {row['model_name']}: {row['accuracy']:.4f} accuracy")
        analysis.append("")

        # Speed analysis
        analysis.append("TRAINING SPEED ANALYSIS:")
        analysis.append("-" * 25)
        fastest = results_df.loc[results_df['training_time'].idxmin()]
        slowest = results_df.loc[results_df['training_time'].idxmax()]
        analysis.append(f"Fastest: {fastest['model_name']} ({fastest['training_time']:.4f}s)")
        analysis.append(f"Slowest: {slowest['model_name']} ({slowest['training_time']:.4f}s)")
        analysis.append("")

        # Recommendations
        analysis.append("RECOMMENDATIONS:")
        analysis.append("-" * 15)
        best_overall = results_df.loc[results_df['accuracy'].idxmax()]
        analysis.append(f"• Best overall model: {best_overall['model_name']}")
        analysis.append(f"• For production use: Consider {best_overall['model_name']} (accuracy: {best_overall['accuracy']:.4f})")

        fast_and_good = results_df[(results_df['accuracy'] >= 0.95) & (results_df['training_time'] <= 0.01)]
        if not fast_and_good.empty:
            best_fast = fast_and_good.loc[fast_and_good['accuracy'].idxmax()]
            analysis.append(f"• For real-time applications: {best_fast['model_name']} (fast and accurate)")

        return "\n".join(analysis)

    def create_summary_report(self, experiment_name: str,
                            results_summary: Dict[str, Any]) -> Path:
        """Create a comprehensive summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.base_dir / "reports" / "summary" / f"{experiment_name}_summary_{timestamp}.md"

        report_content = f"""# MLflow Experiment Summary Report

**Experiment:** {experiment_name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
{results_summary.get('overview', 'No overview provided')}

## Key Results
{self._format_key_results(results_summary.get('key_results', {}))}

## Recommendations
{self._format_recommendations(results_summary.get('recommendations', []))}

## Files Generated
{self._format_file_list(results_summary.get('files', []))}

---
*Report generated automatically by MLflow Tutorial Output Manager*
"""

        with open(report_file, 'w') as f:
            f.write(report_content)

        self.logger.info(f"Created summary report: {report_file}")
        return report_file

    def _format_key_results(self, results: Dict[str, Any]) -> str:
        """Format key results for markdown"""
        if not results:
            return "No key results provided"

        formatted = []
        for key, value in results.items():
            formatted.append(f"- **{key}**: {value}")

        return "\n".join(formatted)

    def _format_recommendations(self, recommendations: list) -> str:
        """Format recommendations for markdown"""
        if not recommendations:
            return "No recommendations provided"

        formatted = []
        for i, rec in enumerate(recommendations, 1):
            formatted.append(f"{i}. {rec}")

        return "\n".join(formatted)

    def _format_file_list(self, files: list) -> str:
        """Format file list for markdown"""
        if not files:
            return "No files listed"

        formatted = []
        for file_path in files:
            formatted.append(f"- `{file_path}`")

        return "\n".join(formatted)

    def cleanup_old_outputs(self, days_old: int = 7):
        """Clean up output files older than specified days"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)

        cleaned_count = 0
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1

        self.logger.info(f"Cleaned up {cleaned_count} old output files")
        return cleaned_count