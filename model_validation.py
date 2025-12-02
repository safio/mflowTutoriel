#!/usr/bin/env python3
"""
Model Validation & A/B Testing

Comprehensive model validation with:
- Champion vs Challenger testing
- A/B testing framework
- Statistical significance testing
- Performance comparison and automated decision-making

Usage:
    # Compare Production (champion) vs Staging (challenger)
    python model_validation.py --model-name iris-classifier --champion-stage Production --challenger-stage Staging

    # Compare specific versions
    python model_validation.py --model-name iris-classifier --champion-version 1 --challenger-version 2

    # A/B test with custom test data
    python model_validation.py --model-name iris-classifier --champion-version 1 --challenger-version 2 --test-data data.csv

    # Auto-promote if challenger wins
    python model_validation.py --model-name iris-classifier --auto-promote --confidence 0.95
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mlflow_tutorial.experiments.mlflow_registry_manager import MLflowRegistryManager
from utils.output_manager import OutputManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelValidator:
    """Comprehensive model validation and A/B testing framework"""

    def __init__(self):
        self.client = MlflowClient()
        self.registry_manager = MLflowRegistryManager()
        self.output_manager = OutputManager()
        self.champion_model = None
        self.challenger_model = None
        self.champion_info = {}
        self.challenger_info = {}

    def load_models(
        self,
        model_name: str,
        champion_version: Optional[str] = None,
        champion_stage: Optional[str] = None,
        challenger_version: Optional[str] = None,
        challenger_stage: Optional[str] = None
    ):
        """Load champion and challenger models"""
        logger.info("Loading models for comparison...")

        # Load champion model
        if champion_version:
            champion_uri = f"models:/{model_name}/{champion_version}"
            self.champion_info = {'version': champion_version, 'stage': None}
        elif champion_stage:
            champion_uri = f"models:/{model_name}/{champion_stage}"
            self.champion_info = {'version': None, 'stage': champion_stage}
        else:
            champion_uri = f"models:/{model_name}/Production"
            self.champion_info = {'version': None, 'stage': 'Production'}

        self.champion_model = mlflow.pyfunc.load_model(champion_uri)
        logger.info(f"✅ Champion model loaded")

        # Load challenger model
        if challenger_version:
            challenger_uri = f"models:/{model_name}/{challenger_version}"
            self.challenger_info = {'version': challenger_version, 'stage': None}
        elif challenger_stage:
            challenger_uri = f"models:/{model_name}/{challenger_stage}"
            self.challenger_info = {'version': None, 'stage': challenger_stage}
        else:
            challenger_uri = f"models:/{model_name}/Staging"
            self.challenger_info = {'version': None, 'stage': 'Staging'}

        self.challenger_model = mlflow.pyfunc.load_model(challenger_uri)
        logger.info(f"✅ Challenger model loaded")

        self.champion_info['model_name'] = model_name
        self.challenger_info['model_name'] = model_name

    def load_test_data(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load test data for validation"""
        if data_path:
            logger.info(f"Loading test data from {data_path}")
            data = pd.read_csv(data_path)
            # Assume last column is target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1].values
        else:
            logger.info("Using Iris dataset for testing")
            iris = load_iris()
            X = pd.DataFrame(iris.data, columns=iris.feature_names)
            y = iris.target
            # Use only test split
            _, X, _, y = train_test_split(X, y, test_size=0.3, random_state=42)

        logger.info(f"Test data loaded: {len(X)} samples")
        return X, y

    def evaluate_model(self, model, X: pd.DataFrame, y: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        logger.info(f"Evaluating {model_type} model...")

        # Make predictions
        y_pred = model.predict(X)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }

        # Get classification report
        class_report = classification_report(y, y_pred, output_dict=True, zero_division=0)

        # Get confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)

        return {
            'predictions': y_pred,
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }

    def compare_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Compare champion and challenger models"""
        logger.info("Comparing champion vs challenger...")

        # Evaluate both models
        champion_results = self.evaluate_model(self.champion_model, X, y, "Champion")
        challenger_results = self.evaluate_model(self.challenger_model, X, y, "Challenger")

        # Create comparison DataFrame
        comparison_data = []
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            champion_val = champion_results['metrics'][metric_name]
            challenger_val = challenger_results['metrics'][metric_name]
            diff = challenger_val - champion_val
            pct_change = (diff / champion_val * 100) if champion_val > 0 else 0

            comparison_data.append({
                'metric': metric_name,
                'champion': champion_val,
                'challenger': challenger_val,
                'difference': diff,
                'pct_change': pct_change,
                'winner': 'Challenger' if diff > 0 else ('Champion' if diff < 0 else 'Tie')
            })

        comparison_df = pd.DataFrame(comparison_data)

        return {
            'champion_results': champion_results,
            'challenger_results': challenger_results,
            'comparison': comparison_df
        }

    def statistical_significance_test(
        self,
        champion_preds: np.ndarray,
        challenger_preds: np.ndarray,
        y_true: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform statistical significance test using McNemar's test

        McNemar's test is used to compare paired predictions from two classifiers
        """
        # Create contingency table
        # [both_correct, champion_correct_challenger_wrong]
        # [champion_wrong_challenger_correct, both_wrong]
        both_correct = np.sum((champion_preds == y_true) & (challenger_preds == y_true))
        both_wrong = np.sum((champion_preds != y_true) & (challenger_preds != y_true))
        champion_only_correct = np.sum((champion_preds == y_true) & (challenger_preds != y_true))
        challenger_only_correct = np.sum((champion_preds != y_true) & (challenger_preds == y_true))

        # Perform McNemar's test
        # Test statistic
        b = champion_only_correct
        c = challenger_only_correct

        if b + c > 0:
            chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        else:
            chi2_stat = 0
            p_value = 1.0

        alpha = 1 - confidence
        is_significant = p_value < alpha

        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_level': confidence,
            'contingency_table': {
                'both_correct': both_correct,
                'both_wrong': both_wrong,
                'champion_only_correct': champion_only_correct,
                'challenger_only_correct': challenger_only_correct
            }
        }

    def plot_comparison(self, comparison_results: Dict[str, Any], output_path: Optional[str] = None):
        """Create visualization comparing the models"""
        comparison_df = comparison_results['comparison']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Champion vs Challenger Model Comparison', fontsize=16, fontweight='bold')

        # 1. Metric comparison bar chart
        ax = axes[0, 0]
        x = np.arange(len(comparison_df))
        width = 0.35
        ax.bar(x - width/2, comparison_df['champion'], width, label='Champion', alpha=0.8)
        ax.bar(x + width/2, comparison_df['challenger'], width, label='Challenger', alpha=0.8)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['metric'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Percentage change
        ax = axes[0, 1]
        colors = ['green' if x > 0 else 'red' for x in comparison_df['pct_change']]
        ax.barh(comparison_df['metric'], comparison_df['pct_change'], color=colors, alpha=0.7)
        ax.set_xlabel('Percentage Change (%)')
        ax.set_title('Challenger vs Champion (% Change)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3)

        # 3. Champion confusion matrix
        ax = axes[1, 0]
        sns.heatmap(comparison_results['champion_results']['confusion_matrix'],
                   annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title('Champion Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # 4. Challenger confusion matrix
        ax = axes[1, 1]
        sns.heatmap(comparison_results['challenger_results']['confusion_matrix'],
                   annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
        ax.set_title('Challenger Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_manager.get_visualization_path(
                'comparison_plots',
                f'model_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Comparison plot saved to {output_path}")
        plt.close()

        return output_path

    def print_summary(
        self,
        comparison_results: Dict[str, Any],
        significance_results: Dict[str, Any],
        recommendation: str
    ):
        """Print comprehensive validation summary"""
        print("\n" + "="*80)
        print("MODEL VALIDATION SUMMARY")
        print("="*80)

        print(f"\nChampion Model:")
        print(f"  Name:    {self.champion_info['model_name']}")
        print(f"  Version: {self.champion_info.get('version', 'N/A')}")
        print(f"  Stage:   {self.champion_info.get('stage', 'N/A')}")

        print(f"\nChallenger Model:")
        print(f"  Name:    {self.challenger_info['model_name']}")
        print(f"  Version: {self.challenger_info.get('version', 'N/A')}")
        print(f"  Stage:   {self.challenger_info.get('stage', 'N/A')}")

        print("\n" + "-"*80)
        print("PERFORMANCE COMPARISON")
        print("-"*80)
        print(comparison_results['comparison'].to_string(index=False))

        print("\n" + "-"*80)
        print("STATISTICAL SIGNIFICANCE TEST (McNemar's Test)")
        print("-"*80)
        print(f"Chi-Square Statistic: {significance_results['chi2_statistic']:.4f}")
        print(f"P-Value:              {significance_results['p_value']:.4f}")
        print(f"Confidence Level:     {significance_results['confidence_level']*100:.0f}%")
        print(f"Significant:          {'Yes' if significance_results['is_significant'] else 'No'}")

        print("\n" + "-"*80)
        print("RECOMMENDATION")
        print("-"*80)
        print(recommendation)
        print("="*80)

    def make_recommendation(
        self,
        comparison_results: Dict[str, Any],
        significance_results: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """Make a recommendation based on the validation results"""
        comparison_df = comparison_results['comparison']

        # Count metrics where challenger wins
        challenger_wins = (comparison_df['winner'] == 'Challenger').sum()
        total_metrics = len(comparison_df)

        # Check accuracy specifically
        accuracy_row = comparison_df[comparison_df['metric'] == 'accuracy'].iloc[0]
        accuracy_improved = accuracy_row['difference'] > 0

        # Decision logic
        if significance_results['is_significant']:
            if challenger_wins >= total_metrics * 0.75 and accuracy_improved:
                recommendation = (
                    "🚀 STRONG RECOMMENDATION: PROMOTE CHALLENGER\n"
                    f"   - Challenger wins on {challenger_wins}/{total_metrics} metrics\n"
                    f"   - Accuracy improvement: {accuracy_row['pct_change']:.2f}%\n"
                    f"   - Difference is statistically significant (p={significance_results['p_value']:.4f})\n"
                    "   - Safe to promote to Production"
                )
                should_promote = True
            elif challenger_wins >= total_metrics * 0.5:
                recommendation = (
                    "⚠️  CONDITIONAL RECOMMENDATION: CONSIDER PROMOTION\n"
                    f"   - Challenger wins on {challenger_wins}/{total_metrics} metrics\n"
                    f"   - Difference is statistically significant (p={significance_results['p_value']:.4f})\n"
                    "   - Review detailed metrics before promoting"
                )
                should_promote = False
            else:
                recommendation = (
                    "❌ RECOMMENDATION: KEEP CHAMPION\n"
                    f"   - Champion wins on {total_metrics - challenger_wins}/{total_metrics} metrics\n"
                    "   - Challenger needs further improvement"
                )
                should_promote = False
        else:
            recommendation = (
                "⚠️  NO CLEAR WINNER: INSUFFICIENT EVIDENCE\n"
                f"   - Difference is NOT statistically significant (p={significance_results['p_value']:.4f})\n"
                "   - Models perform similarly - consider other factors (speed, resource usage, etc.)\n"
                "   - Recommend more testing or larger test dataset"
            )
            should_promote = False

        return recommendation, should_promote


def main():
    parser = argparse.ArgumentParser(
        description='Model validation and A/B testing framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--model-name', type=str, required=True, help='Name of the model')

    # Champion model
    champion_group = parser.add_mutually_exclusive_group()
    champion_group.add_argument('--champion-version', type=str, help='Champion model version')
    champion_group.add_argument('--champion-stage', type=str, default='Production',
                                help='Champion model stage (default: Production)')

    # Challenger model
    challenger_group = parser.add_mutually_exclusive_group()
    challenger_group.add_argument('--challenger-version', type=str, help='Challenger model version')
    challenger_group.add_argument('--challenger-stage', type=str, default='Staging',
                                  help='Challenger model stage (default: Staging)')

    # Test data
    parser.add_argument('--test-data', type=str, help='Path to test data CSV file')

    # Options
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level (default: 0.95)')
    parser.add_argument('--auto-promote', action='store_true', help='Auto-promote if challenger wins')
    parser.add_argument('--save-plot', action='store_true', default=True, help='Save comparison plot')

    args = parser.parse_args()

    try:
        # Initialize validator
        validator = ModelValidator()

        # Load models
        validator.load_models(
            model_name=args.model_name,
            champion_version=args.champion_version,
            champion_stage=args.champion_stage if not args.champion_version else None,
            challenger_version=args.challenger_version,
            challenger_stage=args.challenger_stage if not args.challenger_version else None
        )

        # Load test data
        X_test, y_test = validator.load_test_data(args.test_data)

        # Compare models
        comparison_results = validator.compare_models(X_test, y_test)

        # Statistical significance test
        significance_results = validator.statistical_significance_test(
            comparison_results['champion_results']['predictions'],
            comparison_results['challenger_results']['predictions'],
            y_test,
            confidence=args.confidence
        )

        # Make recommendation
        recommendation, should_promote = validator.make_recommendation(
            comparison_results,
            significance_results
        )

        # Print summary
        validator.print_summary(comparison_results, significance_results, recommendation)

        # Save plot
        if args.save_plot:
            validator.plot_comparison(comparison_results)

        # Auto-promote if requested and recommended
        if args.auto_promote and should_promote:
            print("\n🚀 Auto-promoting challenger to Production...")
            challenger_version = args.challenger_version or validator.challenger_info.get('version')
            if challenger_version:
                validator.registry_manager.promote_model(
                    model_name=args.model_name,
                    version=challenger_version,
                    to_stage='Production',
                    archive_existing=True
                )
                print("✅ Promotion complete!")
            else:
                print("❌ Cannot auto-promote: challenger version not determined")

        logger.info("✅ Validation completed successfully")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
