"""Data loading utilities for fall detection project."""

import os
import glob
import pandas as pd
import pickle
from typing import List, Dict, Any


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load and organize sensor data from Excel files by trials."""
    print("Loading trial-based data...")
    all_trials = []
    
    subject_folders = glob.glob(os.path.join(data_path, 'sub*'))
    if not subject_folders:
        raise ValueError(f"No subject folders found in {data_path}")

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        
        for trial_type in ['ADLs', 'Falls', 'Near_Falls']:
            trial_folder = os.path.join(subject_folder, trial_type)
            if os.path.exists(trial_folder):
                excel_files = glob.glob(os.path.join(trial_folder, '*.xlsx'))
                
                for file_path in excel_files:
                    try:
                        df = pd.read_excel(file_path)
                        trial_info = {
                            'data': df,
                            'subject_id': subject_id,
                            'trial_type': trial_type,
                            'file_name': os.path.basename(file_path),
                            'file_path': file_path
                        }
                        all_trials.append(trial_info)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    return all_trials


def check_cached(cache_dir: str) -> bool:
    """Check if cached data files exist."""
    required_files = ['X.pkl', 'y.pkl', 'subs.pkl']
    for filename in required_files:
        if not os.path.exists(os.path.join(cache_dir, filename)):
            return False
    return True


def load_cached_data(cache_dir: str):
    """Load cached preprocessed data."""
    with open(f'{cache_dir}/X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open(f'{cache_dir}/y.pkl', 'rb') as f:
        y = pickle.load(f)
    with open(f'{cache_dir}/subs.pkl', 'rb') as f:
        subs = pickle.load(f)
    return X, y, subs


def save_cached_data(cache_dir: str, X, y, subs):
    """Save processed data to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    
    with open(f'{cache_dir}/X.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open(f'{cache_dir}/y.pkl', 'wb') as f:
        pickle.dump(y, f)
    with open(f'{cache_dir}/subs.pkl', 'wb') as f:
        pickle.dump(subs, f)
