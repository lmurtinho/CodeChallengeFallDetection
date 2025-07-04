#!/usr/bin/env python3
"""
Main training script for fall detection models.
"""

import os
import sys
import yaml
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import load_data, check_cached, load_cached_data, save_cached_data
from features.engineering import engineer_features
from features.statistical import create_statistical_features
from data.preprocessor import subject_wise_split, normalize_features
from models.trainer import train_statistical_approach
from models.evaluator import print_results_summary


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def save_models(models_dict: dict, scaler: StandardScaler, config: dict, models_dir: str = "models"):
    """
    Save trained models and scaler to the models directory.
    
    Args:
        models_dict: Dictionary containing trained models
        scaler: Fitted StandardScaler object
        config: Configuration dictionary
        models_dir: Directory to save models (default: "models")
    """
    # Create models directory if it doesn't exist
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save each model
    for model_name, model_data in models_dict.items():
        if 'model' in model_data:
            model_filename = f"{model_name}_{timestamp}.pkl"
            model_path = models_path / model_filename
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data['model'], f)
            
            print(f"Saved {model_name} to {model_path}")
    
    # Save the scaler
    scaler_filename = f"scaler_{timestamp}.pkl"
    scaler_path = models_path / scaler_filename
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Saved scaler to {scaler_path}")
    
    # Save metadata about this training run
    metadata = {
        'timestamp': timestamp,
        'config': config,
        'models_saved': list(models_dict.keys()),
        'scaler_file': scaler_filename
    }
    
    metadata_filename = f"training_metadata_{timestamp}.yaml"
    metadata_path = models_path / metadata_filename
    
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"Saved training metadata to {metadata_path}")
    
    # Create or update a "latest" symlink/copy for easy access
    try:
        # Save the latest model info for easy loading
        latest_info = {
            'timestamp': timestamp,
            'models': {name: f"{name}_{timestamp}.pkl" for name in models_dict.keys()},
            'scaler': scaler_filename,
            'metadata': metadata_filename
        }
        
        latest_path = models_path / "latest_models.yaml"
        with open(latest_path, 'w') as f:
            yaml.dump(latest_info, f, default_flow_style=False)
        
        print(f"Updated latest models info at {latest_path}")
        
    except Exception as e:
        print(f"Warning: Could not create latest models info: {e}")


def main():
    """Main training pipeline."""
    # Load configuration
    config = load_config()
    
    data_path = config['data']['data_path']
    cache_dir = config['data']['cache_dir']
    
    # Check if cached data exists
    data_cached = check_cached(cache_dir) if cache_dir else False
    
    if data_cached:
        X, y, subs = load_cached_data(cache_dir)
    else:
        # Load raw data
        all_trials = load_data(data_path)
        
        # Feature engineering
        print("Engineering features...")
        for trial in tqdm(all_trials):
            trial['processed_data'] = engineer_features(
                trial['data'],
                config['data']['sensor_locations'],
                config['data']['features']
            )
        
        # Statistical feature extraction
        print("Extracting statistical features...")
        for trial in tqdm(all_trials):
            trial['stat_data'] = create_statistical_features(trial['processed_data'])
        
        # Combine all statistical data
        X = pd.concat([trial['stat_data'] for trial in all_trials]).reset_index(drop=True)
        y = pd.Series([1 if trial['trial_type'] == 'Falls' else 0 for trial in all_trials])
        subs = pd.Series([trial['subject_id'] for trial in all_trials])
        
        # Save to cache
        if cache_dir:
            save_cached_data(cache_dir, X, y, subs)
    
    # Split data by subjects
    train_indices, val_indices, test_indices = subject_wise_split(
        subs, 
        config['preprocessing']['test_size'],
        config['preprocessing']['val_size'],
        config['preprocessing']['random_state']
    )
    
    X_train = X.loc[train_indices]
    X_val = X.loc[val_indices]
    X_test = X.loc[test_indices]
    
    y_train = y.loc[train_indices]
    y_val = y.loc[val_indices]
    y_test = y.loc[test_indices]
    
    subs_train = subs.loc[train_indices]
    subs_val = subs.loc[val_indices]
    subs_test = subs.loc[test_indices]
    
    print(f'Train subjects: {sorted(subs_train.unique())}')
    print(f'Validation subjects: {sorted(subs_val.unique())}')
    print(f'Test subjects: {sorted(subs_test.unique())}')
    
    # Normalize features
    scaler = StandardScaler()
    X_train_norm, X_val_norm, X_test_norm = normalize_features(
        X_train, X_val, X_test, scaler
    )
    X_train_norm = pd.DataFrame(X_train_norm, columns=X.columns)
    X_val_norm = pd.DataFrame(X_val_norm, columns=X.columns)
    X_test_norm = pd.DataFrame(X_test_norm, columns=X.columns)
    
    # Train models
    print("\nTraining models...")
    results = train_statistical_approach(X_train_norm, y_train, X_test_norm, y_test)
    
    # Print results
    print_results_summary(results)
    
    # Save models and scaler
    print("\nSaving models...")
    save_models(results, scaler, config)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()