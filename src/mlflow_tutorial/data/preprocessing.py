"""
Data Preprocessing Module

Advanced preprocessing utilities for the iris dataset and general ML preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Advanced data preprocessing for ML experiments."""

    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.is_fitted = False

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        method: str = 'standard'
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Scale features using specified method.

        Args:
            X_train: Training features
            X_test: Test features (optional)
            method: Scaling method ('standard' or 'minmax')

        Returns:
            Scaled features
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")

        # Fit on training data and transform
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        logger.info(f"Applied {method} scaling to features")

        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: Optional[pd.DataFrame] = None,
        k: int = 3
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Select top k features using univariate statistical tests.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            k: Number of features to select

        Returns:
            Selected features
        """
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)

        # Fit on training data and transform
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)

        # Get selected feature names
        selected_features = X_train.columns[self.feature_selector.get_support()].tolist()

        X_train_selected_df = pd.DataFrame(
            X_train_selected,
            columns=selected_features,
            index=X_train.index
        )

        logger.info(f"Selected top {k} features: {selected_features}")

        if X_test is not None:
            X_test_selected = self.feature_selector.transform(X_test)
            X_test_selected_df = pd.DataFrame(
                X_test_selected,
                columns=selected_features,
                index=X_test.index
            )
            return X_train_selected_df, X_test_selected_df

        return X_train_selected_df

    def encode_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Encode string targets to integers (if needed).

        Args:
            y: Target array

        Returns:
            Encoded targets
        """
        if y.dtype == 'object' or isinstance(y[0], str):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            logger.info("Encoded string targets to integers")
            return y_encoded

        return y

    def get_feature_importance_scores(self) -> Optional[dict]:
        """
        Get feature importance scores from the feature selector.

        Returns:
            Dictionary of feature names and their scores
        """
        if self.feature_selector is None:
            logger.warning("Feature selector not fitted yet")
            return None

        feature_names = self.feature_selector.feature_names_in_
        scores = self.feature_selector.scores_

        return dict(zip(feature_names, scores))

    def create_polynomial_features(
        self,
        X: pd.DataFrame,
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features.

        Args:
            X: Input features
            degree: Polynomial degree
            include_bias: Whether to include bias column

        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)

        # Create feature names
        feature_names = poly.get_feature_names_out(X.columns)

        X_poly_df = pd.DataFrame(
            X_poly,
            columns=feature_names,
            index=X.index
        )

        logger.info(f"Created polynomial features of degree {degree}")
        logger.info(f"Features increased from {X.shape[1]} to {X_poly_df.shape[1]}")

        return X_poly_df

    def handle_missing_values(
        self,
        X: pd.DataFrame,
        strategy: str = 'mean'
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            X: Features with potential missing values
            strategy: Imputation strategy ('mean', 'median', 'mode')

        Returns:
            DataFrame with imputed values
        """
        from sklearn.impute import SimpleImputer

        if strategy in ['mean', 'median']:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")

        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Imputed {missing_count} missing values using {strategy} strategy")

        return X_imputed

    def get_preprocessing_summary(self) -> dict:
        """
        Get summary of applied preprocessing steps.

        Returns:
            Dictionary with preprocessing information
        """
        summary = {
            'scaler_applied': self.scaler is not None,
            'feature_selection_applied': self.feature_selector is not None,
            'label_encoding_applied': self.label_encoder is not None
        }

        if self.scaler:
            summary['scaler_type'] = type(self.scaler).__name__

        if self.feature_selector:
            summary['n_features_selected'] = self.feature_selector.k
            summary['feature_scores'] = self.get_feature_importance_scores()

        return summary