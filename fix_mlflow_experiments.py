#!/usr/bin/env python3
"""
Script to fix MLflow experiments by permanently deleting old deleted experiments.
This allows MLflow to create fresh experiments when scripts run.

Usage:
    python fix_mlflow_experiments.py          # Interactive mode (asks for confirmation)
    python fix_mlflow_experiments.py --auto   # Automatic mode (no confirmation)
    python fix_mlflow_experiments.py --check  # Check only (no cleanup)
"""

import mlflow
from mlflow.tracking import MlflowClient
import argparse
import shutil
from pathlib import Path
import sys

def check_deleted_experiments(client):
    """Check for deleted experiments"""
    all_experiments = client.search_experiments(view_type=3)  # 3 = ALL
    deleted_experiments = [exp for exp in all_experiments if exp.lifecycle_stage == 'deleted']
    return deleted_experiments

def cleanup_trash_directory():
    """Remove the MLflow .trash directory"""
    trash_path = Path("mlruns/.trash")

    if not trash_path.exists():
        return False, "Trash directory does not exist"

    try:
        shutil.rmtree(trash_path)
        return True, f"Successfully removed {trash_path}"
    except Exception as e:
        return False, f"Error removing trash directory: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="Fix MLflow experiments by removing deleted experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_mlflow_experiments.py          # Interactive mode
  python fix_mlflow_experiments.py --auto   # Automatic cleanup
  python fix_mlflow_experiments.py --check  # Check only
        """
    )
    parser.add_argument('--auto', action='store_true',
                       help='Run in automatic mode without confirmation')
    parser.add_argument('--check', action='store_true',
                       help='Only check for deleted experiments, do not clean up')

    args = parser.parse_args()

    print("🔧 MLflow Experiment Cleanup Utility")
    print("=" * 60)

    try:
        client = MlflowClient()
    except Exception as e:
        print(f"❌ Error connecting to MLflow: {e}")
        print("\nMake sure MLflow is properly installed:")
        print("  pip install mlflow")
        sys.exit(1)

    # Check for deleted experiments
    deleted_experiments = check_deleted_experiments(client)

    if not deleted_experiments:
        print("✅ No deleted experiments found. Nothing to fix!")
        print("\nYour MLflow tracking is clean.")
        return 0

    print(f"\n📋 Found {len(deleted_experiments)} deleted experiments:")
    for exp in deleted_experiments:
        print(f"   - {exp.name} (ID: {exp.experiment_id})")

    # Check only mode
    if args.check:
        print("\n⚠️  Run without --check flag to clean up these experiments")
        return 0

    print(f"\n⚠️  These experiments will be permanently deleted from MLflow.")
    print("This will allow your scripts to create fresh experiments with the same names.")

    # Ask for confirmation if not in auto mode
    if not args.auto:
        try:
            response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Operation cancelled.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Operation cancelled.")
            return 1

    # Perform cleanup
    print("\n🗑️  Permanently deleting experiments...")

    success, message = cleanup_trash_directory()

    if success:
        print(f"✅ {message}")
        print("\n✅ Cleanup completed successfully!")
        print("\n📝 NEXT STEPS:")
        print("1. Your scripts will now create fresh experiments when run")
        print("2. You can verify with: python -c \"import mlflow; [print(exp.name) for exp in mlflow.MlflowClient().search_experiments(view_type=1)]\"")
        print("\nExample:")
        print("  python train.py --n_estimators 50")
        return 0
    else:
        print(f"❌ {message}")
        print("\n📝 MANUAL CLEANUP:")
        print("If automatic cleanup failed, you can manually run:")
        print("  rm -rf mlruns/.trash")
        return 1

if __name__ == "__main__":
    sys.exit(main())
