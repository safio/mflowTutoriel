#!/usr/bin/env python3
"""
Model Registration Script

Register trained models from MLflow runs to the MLflow Model Registry.
Supports registration, staging transitions, and automatic promotion.

Usage:
    # Register a model from a run
    python register_model.py --run-id <run_id> --model-name iris-classifier

    # Register and transition to staging
    python register_model.py --run-id <run_id> --model-name iris-classifier --stage Staging

    # List all registered models
    python register_model.py --list-models

    # Get details of a specific model
    python register_model.py --model-name iris-classifier --details

    # Promote a model from Staging to Production
    python register_model.py --model-name iris-classifier --promote --version 1

    # Auto-promote based on metrics
    python register_model.py --model-name iris-classifier --auto-promote --version 2 --metric accuracy --threshold 0.95
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mlflow_tutorial.experiments.mlflow_registry_manager import MLflowRegistryManager
import mlflow


def register_model_from_run(args):
    """Register a model from an MLflow run"""
    registry = MLflowRegistryManager()

    tags = {}
    if args.tags:
        for tag in args.tags:
            key, value = tag.split('=')
            tags[key] = value

    version = registry.register_model(
        run_id=args.run_id,
        model_name=args.model_name,
        artifact_path=args.artifact_path,
        tags=tags if tags else None,
        description=args.description
    )

    print(f"\n✅ Model '{args.model_name}' registered as version {version}")

    # Transition to stage if specified
    if args.stage and args.stage != 'None':
        registry.transition_model_stage(
            model_name=args.model_name,
            version=version,
            stage=args.stage
        )
        print(f"✅ Transitioned to '{args.stage}' stage")


def list_all_models(args):
    """List all registered models"""
    registry = MLflowRegistryManager()

    print("\n" + "="*80)
    print("REGISTERED MODELS")
    print("="*80)

    models_df = registry.list_registered_models()

    if models_df.empty:
        print("\nNo registered models found.")
        print("\nTo register a model, use:")
        print("  python register_model.py --run-id <run_id> --model-name <model_name>")
    else:
        print(models_df.to_string(index=False))

    print("\n" + "="*80)


def show_model_details(args):
    """Show details of a specific model"""
    registry = MLflowRegistryManager()

    print("\n" + "="*80)
    print(f"MODEL DETAILS: {args.model_name}")
    print("="*80)

    # Get version details
    details_df = registry.get_model_version_details(
        model_name=args.model_name,
        version=args.version
    )

    if details_df.empty:
        print(f"\n❌ No versions found for model '{args.model_name}'")
    else:
        print("\nVersion Information:")
        print(details_df.to_string(index=False))

    print("\n" + "="*80)


def promote_model(args):
    """Promote a model to the next stage"""
    registry = MLflowRegistryManager()

    if not args.version:
        print("❌ Error: --version required for promotion")
        return

    registry.promote_model(
        model_name=args.model_name,
        version=args.version,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        archive_existing=args.archive_existing
    )

    print(f"\n✅ Successfully promoted '{args.model_name}' v{args.version}")
    print(f"   {args.from_stage} → {args.to_stage}")


def auto_promote_model(args):
    """Automatically promote a model based on metrics"""
    registry = MLflowRegistryManager()

    if not args.version:
        print("❌ Error: --version required for auto-promotion")
        return

    success = registry.auto_promote_model(
        model_name=args.model_name,
        staging_version=args.version,
        metric_name=args.metric,
        metric_threshold=args.threshold,
        compare_to_production=not args.no_compare,
        higher_is_better=not args.lower_is_better
    )

    if success:
        print(f"\n✅ Model auto-promoted to Production")
    else:
        print(f"\n❌ Model did not meet promotion criteria")


def compare_versions(args):
    """Compare two model versions"""
    registry = MLflowRegistryManager()

    if not args.version or not args.compare_version:
        print("❌ Error: --version and --compare-version required")
        return

    print("\n" + "="*80)
    print(f"COMPARING VERSIONS: {args.version} vs {args.compare_version}")
    print("="*80)

    comparison_df = registry.compare_model_versions(
        model_name=args.model_name,
        version1=args.version,
        version2=args.compare_version,
        metric_keys=args.metrics if args.metrics else None
    )

    print("\n", comparison_df.to_string(index=False))
    print("\n" + "="*80)


def rollback_model(args):
    """Rollback to a previous model version"""
    registry = MLflowRegistryManager()

    if not args.version:
        print("❌ Error: --version required for rollback")
        return

    print(f"\n⚠️  WARNING: This will rollback '{args.model_name}' to version {args.version}")

    if not args.yes:
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Rollback cancelled")
            return

    registry.rollback_model(
        model_name=args.model_name,
        target_version=args.version,
        stage=args.to_stage
    )

    print(f"✅ Rollback complete")


def main():
    parser = argparse.ArgumentParser(
        description='MLflow Model Registry Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Common arguments
    parser.add_argument('--model-name', type=str, help='Name of the model')

    # Registration arguments
    parser.add_argument('--run-id', type=str, help='MLflow run ID to register')
    parser.add_argument('--artifact-path', type=str, default='model', help='Artifact path in run (default: model)')
    parser.add_argument('--tags', nargs='*', help='Tags in key=value format')
    parser.add_argument('--description', type=str, help='Model description')

    # Stage management
    parser.add_argument('--stage', type=str, choices=['None', 'Staging', 'Production', 'Archived'],
                        help='Stage to transition to after registration')
    parser.add_argument('--version', type=str, help='Model version')

    # Actions
    parser.add_argument('--list-models', action='store_true', help='List all registered models')
    parser.add_argument('--details', action='store_true', help='Show model details')
    parser.add_argument('--promote', action='store_true', help='Promote model to next stage')
    parser.add_argument('--auto-promote', action='store_true', help='Auto-promote based on metrics')
    parser.add_argument('--compare', action='store_true', help='Compare two model versions')
    parser.add_argument('--rollback', action='store_true', help='Rollback to previous version')

    # Promotion options
    parser.add_argument('--from-stage', type=str, default='Staging', help='Source stage (default: Staging)')
    parser.add_argument('--to-stage', type=str, default='Production', help='Target stage (default: Production)')
    parser.add_argument('--archive-existing', action='store_true', default=True, help='Archive existing versions')
    parser.add_argument('--no-archive', dest='archive_existing', action='store_false', help='Do not archive existing')

    # Auto-promotion options
    parser.add_argument('--metric', type=str, default='accuracy', help='Metric for auto-promotion (default: accuracy)')
    parser.add_argument('--threshold', type=float, help='Minimum metric threshold')
    parser.add_argument('--no-compare', action='store_true', help='Do not compare to production')
    parser.add_argument('--lower-is-better', action='store_true', help='Lower metric values are better')

    # Comparison options
    parser.add_argument('--compare-version', type=str, help='Second version to compare')
    parser.add_argument('--metrics', nargs='*', help='Specific metrics to compare')

    # Confirmation
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompts')

    args = parser.parse_args()

    try:
        # Route to appropriate action
        if args.list_models:
            list_all_models(args)

        elif args.details:
            if not args.model_name:
                print("❌ Error: --model-name required")
                return
            show_model_details(args)

        elif args.compare:
            if not args.model_name:
                print("❌ Error: --model-name required")
                return
            compare_versions(args)

        elif args.promote:
            if not args.model_name:
                print("❌ Error: --model-name required")
                return
            promote_model(args)

        elif args.auto_promote:
            if not args.model_name:
                print("❌ Error: --model-name required")
                return
            auto_promote_model(args)

        elif args.rollback:
            if not args.model_name:
                print("❌ Error: --model-name required")
                return
            rollback_model(args)

        elif args.run_id and args.model_name:
            register_model_from_run(args)

        else:
            parser.print_help()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
