# src/__init__.py
"""
Fall Detection Package

A machine learning solution for fall detection using body sensor data.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# src/data/__init__.py
"""
Data handling modules for fall detection.
"""

from .loader import load_data, check_cached, load_cached_data, save_cached_data
from .preprocessor import subject_wise_split, normalize_features

__all__ = [
    'load_data',
    'check_cached',
    'load_cached_data', 
    'save_cached_data',
    'subject_wise_split',
    'normalize_features'
]

# src/features/__init__.py
"""
Feature extraction and engineering modules.
"""

from .engineering import calculate_magnitude, engineer_features
from .statistical import create_statistical_features

__all__ = [
    'calculate_magnitude',
    'engineer_features',
    'create_statistical_features'
]

# src/models/__init__.py
"""
Model training and evaluation modules.
"""

from .trainer import get_model_configs, train_single_model, train_statistical_approach
from .evaluator import evaluate_model, print_results_summary

__all__ = [
    'get_model_configs',
    'train_single_model',
    'train_statistical_approach',
    'evaluate_model',
    'print_results_summary'
]

# src/utils/__init__.py
"""
Utility functions.
"""

# tests/__init__.py
"""
Test modules for fall detection package.
"""
