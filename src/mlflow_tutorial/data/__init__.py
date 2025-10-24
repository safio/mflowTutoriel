"""
Data Management Module

Handles data loading, preprocessing, and preparation for ML experiments.
"""

from .data_manager import DataManager
from .preprocessing import DataPreprocessor

__all__ = ['DataManager', 'DataPreprocessor']