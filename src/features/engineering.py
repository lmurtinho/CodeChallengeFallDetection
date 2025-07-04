"""Feature engineering utilities for sensor data."""

import numpy as np
import pandas as pd
from typing import List


def calculate_magnitude(data: pd.DataFrame, axis_cols: List[str]) -> pd.Series:
    """Calculate the magnitude of a 3D vector."""
    return np.sqrt(
        data[axis_cols[0]]**2 +
        data[axis_cols[1]]**2 +
        data[axis_cols[2]]**2
    )


def engineer_features(data: pd.DataFrame, sensor_locations: List[str], features: List[str]) -> pd.DataFrame:
    """Create engineered features (magnitude) from raw sensor data."""
    for location in sensor_locations:
        for feature in features:
            cols = [col for col in data.columns if location in col and feature in col]
            if len(cols) == 3:
                data[f'{location}_{feature}_magnitude'] = calculate_magnitude(data, cols)
    return data
