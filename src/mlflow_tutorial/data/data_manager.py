"""
Data Manager for MLflow Tutorial

Handles data loading, splitting, and basic preprocessing for the iris dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading and preprocessing for the iris classification tutorial."""

    def __init__(self):
        self.iris_data = None
        self.feature_names = None
        self.target_names = None
        self._load_iris_data()

    def _load_iris_data(self):
        """Load the iris dataset."""
        logger.info("Loading iris dataset...")
        self.iris_data = load_iris()
        self.feature_names = self.iris_data.feature_names
        self.target_names = self.iris_data.target_names
        logger.info(f"Loaded iris dataset with {len(self.iris_data.data)} samples")

    def get_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Get the full iris dataset.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Features and targets
        """
        X = pd.DataFrame(self.iris_data.data, columns=self.feature_names)
        y = self.iris_data.target
        return X, y

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Get train/test split of the iris dataset.

        Args:
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility
            stratify: Whether to stratify the split

        Returns:
            Tuple containing X_train, X_test, y_train, y_test
        """
        X, y = self.get_data()

        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def get_feature_info(self) -> dict:
        """
        Get information about the features.

        Returns:
            dict: Feature information including names, statistics
        """
        X, y = self.get_data()

        info = {
            'feature_names': list(self.feature_names),
            'target_names': list(self.target_names),
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'n_classes': len(self.target_names),
            'feature_stats': X.describe().to_dict(),
            'class_distribution': pd.Series(y).value_counts().to_dict()
        }

        return info

    def get_sample_data(self, n_samples: int = 5) -> pd.DataFrame:
        """
        Get a sample of the data for testing purposes.

        Args:
            n_samples: Number of samples to return

        Returns:
            pd.DataFrame: Sample data with features
        """
        X, _ = self.get_data()
        return X.sample(n=min(n_samples, len(X)), random_state=42)

    def save_sample_data(self, filepath: str, n_samples: int = 10):
        """
        Save sample data to CSV file for testing.

        Args:
            filepath: Path to save the CSV file
            n_samples: Number of samples to save
        """
        sample_data = self.get_sample_data(n_samples)
        sample_data.to_csv(filepath, index=False)
        logger.info(f"Saved {len(sample_data)} samples to {filepath}")