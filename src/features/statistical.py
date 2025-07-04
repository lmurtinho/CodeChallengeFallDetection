"""Statistical feature extraction from time series data."""

import numpy as np
import pandas as pd


def create_statistical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical features for each column."""
    stats = {}
    
    for col in data.columns:
        if col == 'Time':
            continue
            
        stats[f'{col}_mean'] = np.mean(data[col])
        stats[f'{col}_std'] = np.std(data[col])
        stats[f'{col}_min'] = np.min(data[col])
        stats[f'{col}_max'] = np.max(data[col])
        stats[f'{col}_median'] = np.median(data[col])
        stats[f'{col}_var'] = np.var(data[col])
    
    return pd.DataFrame([stats])
