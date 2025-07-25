"""
Model evaluation utilities for fall detection.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from typing import Dict, Any


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
    """
    Evaluate a trained model and return comprehensive metrics.

    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display

    Returns:
        Dictionary containing all evaluation metrics
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f"\n{model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"{model_name} F1 score: {f1_score(y_test, predictions):.4f}")

    # Calculate ROC AUC if probabilities are available
    roc_auc = None
    if probabilities is not None:
        try:
            roc_auc = roc_auc_score(y_test, probabilities)
        except ValueError:
            # Handle case where there's only one class in y_test
            roc_auc = None

    result = {
        'model': model,
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'predictions': predictions,
        'probabilities': probabilities,
        'confusion_matrix': confusion_matrix(y_test, predictions)
    }
    
    if roc_auc is not None:
        result['roc_auc'] = roc_auc
    
    return result


def print_results_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a summary of all model results.

    Args:
        results: Dictionary containing results for each model
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)

    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")