#!/usr/bin/env python3
"""
Model evaluation script for fall detection models.
"""

import numpy as np
import os
import sys
import yaml
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import load_cached_data
from data.preprocessor import subject_wise_split, normalize_features
from models.evaluator import evaluate_model, print_results_summary


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_latest_models_info(models_dir: str) -> dict:
    """Load information about the latest saved models."""
    latest_info_path = Path(models_dir) / "latest_models.yaml"
    
    if not latest_info_path.exists():
        raise FileNotFoundError(f"No latest_models.yaml found in {models_dir}. Please train models first.")
    
    with open(latest_info_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_scaler(models_dir: str, latest_info: dict = None):
    """Load the latest trained models and scaler."""
    if latest_info is None:
        latest_info = load_latest_models_info(models_dir)
    
    models_path = Path(models_dir)
    loaded_models = {}
    
    # Load each model
    for model_name, model_filename in latest_info['models'].items():
        model_path = models_path / model_filename
        
        if not model_path.exists():
            print(f"Warning: Model file {model_path} not found, skipping {model_name}")
            continue
            
        with open(model_path, 'rb') as f:
            loaded_models[model_name] = pickle.load(f)
        
        print(f"Loaded {model_name} from {model_path}")
    
    # Load scaler
    scaler_path = models_path / latest_info['scaler']
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file {scaler_path} not found")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Loaded scaler from {scaler_path}")
    
    return loaded_models, scaler


def load_training_metadata(models_dir: str, latest_info: dict = None):
    """Load training metadata for the latest models."""
    if latest_info is None:
        latest_info = load_latest_models_info(models_dir)
    
    metadata_path = Path(models_dir) / latest_info['metadata']
    
    if not metadata_path.exists():
        print(f"Warning: Metadata file {metadata_path} not found")
        return None
    
    with open(metadata_path, 'r') as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_names, model_name: str, top_n: int = 20, save_path: str = None):
    """Plot feature importance for tree-based models and logistic regression."""
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = model.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(model, 'coef_'):
        # Linear models (logistic regression, SVM, etc.)
        # Use absolute values of coefficients as importance
        coef = model.coef_
        if coef.ndim > 1:
            # Multi-class case: take the first class or mean across classes
            importance = np.abs(coef[0]) if coef.shape[0] == 1 else np.abs(coef).mean(axis=0)
        else:
            importance = np.abs(coef)
        importance_type = "Absolute Coefficients"
    else:
        print(f"Model {model_name} doesn't have feature importances or coefficients.")
        return None
    
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feature_imp.head(top_n), x='importance', y='feature', ax=ax)
    ax.set_title(f'Top {top_n} {importance_type} - {model_name}')
    ax.set_xlabel(importance_type)
    
    # Adjust layout to prevent label cutoff
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()
    
    return feature_imp


def evaluate_saved_models(models_dir: str, data_cache_dir: str, plots_dir: str = "plots"):
    """Evaluate saved models and generate reports."""
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load latest models info
    latest_info = load_latest_models_info(models_dir)
    print(f"Evaluating models from training run: {latest_info['timestamp']}")
    
    # Load training metadata to get the same configuration
    metadata = load_training_metadata(models_dir, latest_info)
    if metadata:
        config = metadata['config']
        print(f"Using configuration from training run")
    else:
        # Fallback to current config
        config = load_config()
        print("Warning: Using current config file, results may differ from training")
    
    # Load test data
    X, y, subs = load_cached_data(data_cache_dir)
    
    # Use the same split configuration as training
    train_indices, val_indices, test_indices = subject_wise_split(
        subs, 
        config['preprocessing']['test_size'],
        config['preprocessing']['val_size'],
        config['preprocessing']['random_state']
    )
    
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    
    print(f"Test set size: {len(X_test)} samples")
    print(f"Test subjects: {sorted(subs.loc[test_indices].unique())}")
    
    # Load models and scaler
    models, scaler = load_model_and_scaler(models_dir, latest_info)
    
    if not models:
        print("No models loaded successfully")
        return
    
    # Normalize test data using the same scaler from training
    X_train = X.loc[train_indices]
    X_val = X.loc[val_indices]
    
    # We need to fit the scaler on training data and transform test data
    # But since we saved the fitted scaler, we can directly transform
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X.columns,
        index=X_test.index
    )
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test_norm)
        
        # Create result dictionary (matching the format expected by evaluate_model)
        result = {
            'model': model,
            'predictions': y_pred,
            'model_name': model_name
        }
        
        # You might want to add more metrics here if your evaluate_model function returns them
        # For now, we'll compute basic metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        result.update({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        })
        
        results[model_name] = result
        
        # Generate detailed classification report
        print(f"\nDetailed Classification Report - {model_name}:")
        print(classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall']))
        
        # Plot confusion matrix
        cm_path = os.path.join(plots_dir, f"confusion_matrix_{model_name}_{latest_info['timestamp']}.png")
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
        
        # Plot feature importance (if available)
        fi_path = os.path.join(plots_dir, f"feature_importance_{model_name}_{latest_info['timestamp']}.png")
        feature_imp = plot_feature_importance(model, X.columns, model_name, save_path=fi_path)
        
        if feature_imp is not None:
            # Save feature importance to CSV
            fi_csv_path = os.path.join(plots_dir, f"feature_importance_{model_name}_{latest_info['timestamp']}.csv")
            feature_imp.to_csv(fi_csv_path, index=False)
            print(f"Feature importance saved to {fi_csv_path}")
    
    # Print summary
    print_results_summary(results)
    
    # Save evaluation results
    eval_results_path = os.path.join(plots_dir, f"evaluation_results_{latest_info['timestamp']}.yaml")
    eval_summary = {
        'timestamp': latest_info['timestamp'],
        'test_subjects': sorted(subs.loc[test_indices].unique()),
        'test_samples': len(X_test),
        'models_evaluated': list(models.keys()),
        'results': {name: {k: v for k, v in result.items() if k != 'model' and k != 'predictions'} 
                   for name, result in results.items()}
    }
    
    with open(eval_results_path, 'w') as f:
        yaml.dump(eval_summary, f, default_flow_style=False)
    
    print(f"\nEvaluation summary saved to {eval_results_path}")
    
    return results


def evaluate_specific_model(models_dir: str, model_name: str, data_cache_dir: str, plots_dir: str = "plots"):
    """Evaluate a specific model by name."""
    
    # Load latest models info
    latest_info = load_latest_models_info(models_dir)
    
    if model_name not in latest_info['models']:
        print(f"Model '{model_name}' not found in latest models.")
        print(f"Available models: {list(latest_info['models'].keys())}")
        return
    
    # Load only the specific model
    models_path = Path(models_dir)
    model_path = models_path / latest_info['models'][model_name]
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    scaler_path = models_path / latest_info['scaler']
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load and prepare test data (similar to evaluate_saved_models)
    X, y, subs = load_cached_data(data_cache_dir)
    
    # Use saved config if available
    metadata = load_training_metadata(models_dir, latest_info)
    config = metadata['config'] if metadata else load_config()
    
    train_indices, val_indices, test_indices = subject_wise_split(
        subs, 
        config['preprocessing']['test_size'],
        config['preprocessing']['val_size'],
        config['preprocessing']['random_state']
    )
    
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X.columns,
        index=X_test.index
    )
    
    # Evaluate the specific model
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test_norm)
    
    print(f"\nDetailed Classification Report - {model_name}:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall']))
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(plots_dir, f"confusion_matrix_{model_name}_{latest_info['timestamp']}.png")
    plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
    
    # Plot feature importance
    fi_path = os.path.join(plots_dir, f"feature_importance_{model_name}_{latest_info['timestamp']}.png")
    feature_imp = plot_feature_importance(model, X.columns, model_name, save_path=fi_path)
    
    if feature_imp is not None:
        fi_csv_path = os.path.join(plots_dir, f"feature_importance_{model_name}_{latest_info['timestamp']}.csv")
        feature_imp.to_csv(fi_csv_path, index=False)
        print(f"Feature importance saved to {fi_csv_path}")


def main():
    """Main evaluation pipeline."""
    config = load_config()
    
    models_dir = config.get('output', {}).get('models_dir', 'models')
    cache_dir = config['data']['cache_dir']
    plots_dir = config.get('output', {}).get('plots_dir', 'plots')
    
    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} not found. Please train models first.")
        return
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} not found. Please run training first.")
        return
    
    # Check if user wants to evaluate a specific model
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"Evaluating specific model: {model_name}")
        evaluate_specific_model(models_dir, model_name, cache_dir, plots_dir)
    else:
        # Evaluate all saved models
        print("Evaluating all saved models...")
        results = evaluate_saved_models(models_dir, cache_dir, plots_dir)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()