"""
Model training utilities for fall detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List


def get_model_configs() -> Dict[str, Any]:
    """
    Get configuration for different models.
    
    Returns:
        Dictionary of model configurations
    """
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ),
        'Logistic Regression L1': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            penalty='l1',
            solver='liblinear'
        )
    }


def train_single_model(
    model, 
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    model_name: str
) -> Dict[str, Any]:
    """
    Train a single model and evaluate it.
    
    Args:
        model: Untrained sklearn model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
        
    Returns:
        Dictionary containing model and evaluation metrics
    """
    from models.evaluator import evaluate_model
    
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    return evaluate_model(model, X_test, y_test, model_name)


def train_statistical_approach(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple models using statistical features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing results for each model
    """
    results = {}
    model_configs = get_model_configs()

    for model_name, model in model_configs.items():
        results[model_name] = train_single_model(
            model, X_train, y_train, X_test, y_test, model_name
        )

    return results