#!/usr/bin/env python3
"""
Script to clean up old results, logs, and temporary files
"""

import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from utils.output_manager import OutputManager

def clean_old_files(directory: Path, days_old: int = 7, file_patterns: list = None):
    """Clean files older than specified days"""
    if not directory.exists():
        print(f"📁 Directory {directory} does not exist, skipping...")
        return 0

    cutoff_time = datetime.now() - timedelta(days=days_old)
    cleaned_count = 0

    print(f"🧹 Cleaning files older than {days_old} days in {directory}")

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file

            # Check file pattern if specified
            if file_patterns:
                if not any(file_path.match(pattern) for pattern in file_patterns):
                    continue

            # Check file age
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_time:
                try:
                    file_path.unlink()
                    print(f"  🗑️  Deleted: {file_path}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"  ❌ Error deleting {file_path}: {e}")

    return cleaned_count

def clean_empty_directories(directory: Path):
    """Remove empty directories"""
    if not directory.exists():
        return 0

    removed_count = 0

    # Walk bottom-up to handle nested empty directories
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if not any(dir_path.iterdir()):  # Directory is empty
                    dir_path.rmdir()
                    print(f"  📂 Removed empty directory: {dir_path}")
                    removed_count += 1
            except OSError:
                # Directory not empty or other error
                pass

    return removed_count

def clean_temporary_files():
    """Clean temporary files from project root"""
    temp_patterns = [
        "temp_*.png",
        "temp_*.csv",
        "temp_*.json",
        "*.tmp",
        "*.temp",
        "confusion_matrix_*.png",
        "model_comparison*.png",
        "classification_report_*.txt",
        "test_results.csv",
        "model_comparison_results.csv"
    ]

    project_root = Path(".")
    cleaned_count = 0

    print("🧹 Cleaning temporary files from project root...")

    for pattern in temp_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    print(f"  🗑️  Deleted: {file_path}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"  ❌ Error deleting {file_path}: {e}")

    return cleaned_count

def clean_mlflow_artifacts(days_old: int = 30):
    """Clean old MLflow artifacts"""
    mlruns_dir = Path("mlruns")
    cleaned_count = 0

    if not mlruns_dir.exists():
        print("📁 MLflow runs directory does not exist, skipping...")
        return 0

    cutoff_time = datetime.now() - timedelta(days=days_old)

    print(f"🧹 Cleaning MLflow artifacts older than {days_old} days...")

    # Clean old run directories
    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.isdigit():
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir():
                    try:
                        dir_time = datetime.fromtimestamp(run_dir.stat().st_mtime)
                        if dir_time < cutoff_time:
                            shutil.rmtree(run_dir)
                            print(f"  🗑️  Deleted run: {run_dir}")
                            cleaned_count += 1
                    except Exception as e:
                        print(f"  ❌ Error deleting {run_dir}: {e}")

    return cleaned_count

def get_directory_size(directory: Path):
    """Get total size of directory in MB"""
    if not directory.exists():
        return 0

    total_size = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            try:
                total_size += file_path.stat().st_size
            except (OSError, FileNotFoundError):
                pass

    return total_size / (1024 * 1024)  # Convert to MB

def show_cleanup_summary(before_sizes: dict, after_sizes: dict, cleaned_counts: dict):
    """Show summary of cleanup operation"""
    print("\n" + "="*60)
    print("🧹 CLEANUP SUMMARY")
    print("="*60)

    total_files_cleaned = sum(cleaned_counts.values())
    total_space_saved = 0

    for location in before_sizes:
        space_saved = before_sizes[location] - after_sizes[location]
        total_space_saved += space_saved

        print(f"\n📂 {location}:")
        print(f"   Files cleaned: {cleaned_counts.get(location, 0)}")
        print(f"   Space before: {before_sizes[location]:.2f} MB")
        print(f"   Space after:  {after_sizes[location]:.2f} MB")
        print(f"   Space saved:  {space_saved:.2f} MB")

    print(f"\n🎯 TOTAL SUMMARY:")
    print(f"   Total files cleaned: {total_files_cleaned}")
    print(f"   Total space saved: {total_space_saved:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Clean up old results and temporary files")
    parser.add_argument("--days", type=int, default=7,
                       help="Remove files older than N days (default: 7)")
    parser.add_argument("--mlflow-days", type=int, default=30,
                       help="Remove MLflow artifacts older than N days (default: 30)")
    parser.add_argument("--outputs-dir", type=str, default="outputs",
                       help="Outputs directory to clean (default: outputs)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--deep-clean", action="store_true",
                       help="Perform deep cleaning including all temporary files")
    parser.add_argument("--keep-reports", action="store_true",
                       help="Keep reports and summary files during cleanup")

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print("-" * 50)

    # Record sizes before cleanup
    locations = {
        "outputs": Path(args.outputs_dir),
        "mlruns": Path("mlruns"),
        "project_root": Path(".")
    }

    before_sizes = {}
    for name, path in locations.items():
        before_sizes[name] = get_directory_size(path)

    print("🚀 Starting cleanup process...")
    print(f"📅 Cleaning files older than {args.days} days")

    if args.dry_run:
        print("⚠️  DRY RUN: No files will actually be deleted")

    cleaned_counts = {}

    # Clean outputs directory
    outputs_dir = Path(args.outputs_dir)
    if outputs_dir.exists():
        if not args.dry_run:
            # Clean logs
            cleaned_counts["logs"] = clean_old_files(
                outputs_dir / "logs", args.days, ["*.log"]
            )

            # Clean results (but maybe keep some recent ones)
            if not args.keep_reports:
                cleaned_counts["results"] = clean_old_files(
                    outputs_dir / "results", args.days
                )
            else:
                # Only clean very old results
                cleaned_counts["results"] = clean_old_files(
                    outputs_dir / "results", args.days * 2
                )

            # Clean visualizations
            cleaned_counts["visualizations"] = clean_old_files(
                outputs_dir / "visualizations", args.days, ["*.png", "*.jpg", "*.pdf"]
            )

            # Clean models
            cleaned_counts["models"] = clean_old_files(
                outputs_dir / "models", args.days, ["*.pkl", "*.joblib", "*.h5"]
            )

            # Clean empty directories
            cleaned_counts["empty_dirs"] = clean_empty_directories(outputs_dir)
        else:
            print(f"Would clean old files from {outputs_dir}")
            cleaned_counts["outputs"] = 0

    # Clean temporary files
    if not args.dry_run:
        cleaned_counts["temp_files"] = clean_temporary_files()
    else:
        print("Would clean temporary files from project root")
        cleaned_counts["temp_files"] = 0

    # Clean MLflow artifacts
    if not args.dry_run:
        cleaned_counts["mlflow"] = clean_mlflow_artifacts(args.mlflow_days)
    else:
        print(f"Would clean MLflow artifacts older than {args.mlflow_days} days")
        cleaned_counts["mlflow"] = 0

    # Deep clean if requested
    if args.deep_clean:
        print("\n🔥 Performing deep clean...")
        if not args.dry_run:
            # Clean all *.pyc files
            pyc_count = 0
            for pyc_file in Path(".").rglob("*.pyc"):
                pyc_file.unlink()
                pyc_count += 1

            # Clean __pycache__ directories
            cache_count = 0
            for cache_dir in Path(".").rglob("__pycache__"):
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                    cache_count += 1

            cleaned_counts["deep_pyc"] = pyc_count
            cleaned_counts["deep_cache"] = cache_count
            print(f"  🗑️  Cleaned {pyc_count} .pyc files and {cache_count} __pycache__ directories")

    # Record sizes after cleanup
    after_sizes = {}
    for name, path in locations.items():
        after_sizes[name] = get_directory_size(path)

    # Show summary
    if not args.dry_run:
        show_cleanup_summary(before_sizes, after_sizes, cleaned_counts)
    else:
        print("\n🔍 DRY RUN COMPLETE - No files were actually deleted")

    # Recommendation for automation
    if not args.dry_run:
        print(f"\n💡 TIP: You can automate this cleanup by adding to your crontab:")
        print(f"   python cleanup_results.py --days {args.days} --mlflow-days {args.mlflow_days}")

if __name__ == "__main__":
    main()