import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

def calculate_magnitude(data: pd.DataFrame, axis_cols: list[str]) -> pd.Series:
  """
  Calculate the magnitude of a 3D vector from its components provided in a pandas DataFrame.
  Args:
    data: A pandas DataFrame containing the vector component data.
    axis_cols: A list of three strings, the column names in `data` corresponding
               to the x, y, and z components of the vector.
  Returns:
    A pandas Series containing the calculated magnitudes for each row in `data`.
  """
  return np.sqrt(
      data[axis_cols[0]]**2 +
      data[axis_cols[1]]**2 +
      data[axis_cols[2]]**2
  )

def engineer_features(
    data: pd.DataFrame,
    sensor_locations: list[str] = [
        'r.ankle', 'l.ankle', 'r.thigh', 'l.thigh', 'head', 'sternum', 'waist'],
    features: list[str] = ['Acceleration','Angular Velocity', 'Magnetic Field']
    ) -> pd.DataFrame:
    """
    Creates engineered features (magnitude) from raw sensor data.
    Args:
        data: A Pandas DataFrame containing the raw sensor data.
        sensor_locations: A list of sensor locations to process.
        features: A list of feature types to process.
    Returns:
        A Pandas DataFrame with the engineered magnitude features added.
    """
    for location in sensor_locations:
        for feature in features:
          cols = [col for col in data.columns
                  if location in col and feature in col]
          if len(cols) == 3:
            data[f'{location}_{feature}_magnitude'] = calculate_magnitude(data, cols)
    return data

def load_data(data_path):
  """
  Load and organize sensor data from Excel files by trials.
  Args:
      data_path: The path to the root directory containing subject folders.
  Returns:
      A list of dictionaries, where each dictionary represents a trial
      and contains the sensor data (as a pandas DataFrame) along with
      metadata like subject ID, trial type, file name, and file path.
  Raises:
      ValueError: If no subject folders are found in the specified data_path.
  """
  print("Loading trial-based data...")
  all_trials = []

  # Get all subject folders
  subject_folders = glob.glob(os.path.join(data_path, 'sub*'))

  if not subject_folders:
      raise ValueError(f"No subject folders found in {data_path}")

  for subject_folder in tqdm(subject_folders):
      subject_id = os.path.basename(subject_folder)

      # Process each trial type
      for trial_type in ['ADLs', 'Falls', 'Near_Falls']:
          # Check if the trial folder exists
          trial_folder = os.path.join(subject_folder, trial_type)
          if os.path.exists(trial_folder):
              excel_files = glob.glob(os.path.join(trial_folder, '*.xlsx'))

              for file_path in excel_files:
                  try:
                      df = pd.read_excel(file_path)

                      # Store each trial as a separate entity
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

import yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
data_path = config['data']['data_path']
all_trials = load_data(data_path)

for trial in tqdm(all_trials):
  trial['processed_data'] = engineer_features(trial['data'])

def get_magnitude_features(data):
    columns = [col for col in data.columns if (col == 'Time') or ('magnitude' in col)]
    return data[columns]

for trial in tqdm(all_trials):
  trial['processed_data'] = engineer_features(trial['data'])

for trial in tqdm(all_trials):
    trial['magnitude_data'] = get_magnitude_features(trial['processed_data'])

from typing import Tuple
from sklearn.model_selection import train_test_split

def subject_wise_train_test_split(
    subs: pd.Series,
    test_size: float = 0.25,
    val_size: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Splits trial indices into training and testing sets based on unique subjects,
    ensuring that all trials from a given subject are allocated to the same data split.
    
    This is for proper train/test evaluation where test subjects are completely held out.
    
    Args:
        subs: A pandas Series where each element represents the subject ID for a
              corresponding trial.
        test_size: The proportion of subjects to include in the test split.
                   Should be between 0.0 and 1.0.
        random_state: Random seed for reproducibility.
        
    Returns:
        A tuple containing:
        - train_indices: Indices of trials belonging to the training set.
        - test_indices: Indices of trials belonging to the testing set.
        - train_subjects: List of subject IDs in training set.
        - test_subjects: List of subject IDs in testing set.
    """
    # Get and split unique subjects
    subjects = sorted(subs.unique())
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=test_size,
        random_state=random_state
    )
    val_size = val_size / (1 - test_size)
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=val_size,
        random_state=random_state
    )
    
    # Get indices for train and test sets according to subs
    train_indices = subs[subs.isin(train_subjects)].index
    val_indices = subs[subs.isin(val_subjects)].index
    test_indices = subs[subs.isin(test_subjects)].index

    # Shuffle indices
    train_indices = np.random.permutation(train_indices)
    val_indices = np.random.permutation(val_indices)
    test_indices = np.random.permutation(test_indices)

    return train_indices, val_indices, test_indices, train_subjects, val_subjects, test_subjects

subs = pd.Series([i['subject_id'] for i in all_trials])

train_indices, val_indices, test_indices, train_subjects, val_subjects, test_subjects = subject_wise_train_test_split(subs)
print(f'Train subjects: {train_subjects}',
      f'\nValidation subjects: {val_subjects}',
      f'\nTest subjects: {test_subjects}')

# Prepare magnitude data for CNN model - train, validation, and test sets

def prepare_trial_data_full_sequence_magnitude_only(
    all_trials: list,
    max_sequence_length: int = 2562
) -> tuple:
    """
    Prepare magnitude-only trial data with full sequence length and padding.
    
    Args:
        all_trials: List of trial dictionaries with processed_data containing magnitude features
        max_sequence_length: Maximum sequence length for padding (2562)
        
    Returns:
        Tuple of (X, y, subjects) where X contains padded full-length magnitude sequences
    """
    sequences = []
    labels = []
    subjects = []
    
    for trial in all_trials:
        # Get magnitude columns only
        magnitude_cols = [col for col in trial['processed_data'].columns if 'magnitude' in col]
        data = trial['processed_data'][magnitude_cols].values
        
        # Pad or truncate to max_sequence_length
        if len(data) < max_sequence_length:
            # Pad with zeros
            padding = np.zeros((max_sequence_length - len(data), data.shape[1]))
            data = np.vstack([data, padding])
        else:
            # Truncate
            data = data[:max_sequence_length]
        
        sequences.append(data)
        labels.append(1 if trial['trial_type'] == 'Falls' else 0)
        subjects.append(trial['subject_id'])
    
    X = np.array(sequences)
    y = np.array(labels, dtype=int)
    subjects = np.array(subjects)
    
    print(f"Prepared {len(X)} sequences with shape {X.shape}")
    print(f"Magnitude features: {X.shape[2]}")
    print(f"Sequence length: {X.shape[1]}")
    print(f"Labels: {np.bincount(y)} (0=normal, 1=fall)")
    
    return X, y, subjects

X, y, subjects = prepare_trial_data_full_sequence_magnitude_only(all_trials)

# Get indices for training and testing data
train_mask = np.isin(subjects, train_subjects)
val_mask = np.isin(subjects, val_subjects)
test_mask = np.isin(subjects, test_subjects)

X_train_all = X[train_mask]
y_train_all = y[train_mask]
subjects_train = subjects[train_mask]

X_val_final = X[val_mask]
y_val_final = y[val_mask]
subjects_val = subjects[val_mask]

X_test_final = X[test_mask]
y_test_final = y[test_mask]
subjects_test = subjects[test_mask]

print(X_train_all.shape, y_train_all.shape, subjects_train.shape) 
print(X_val_final.shape, y_val_final.shape, subjects_val.shape)
print(X_test_final.shape, y_test_final.shape, subjects_test.shape)

# Save the processed data for later use
import pickle
def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, filename):
    """
    Save processed data to a pickle file.
    
    Args:
        X: Feature data (numpy array).
        y: Labels (numpy array).
        subjects: Subject IDs (numpy array).
        filename: Name of the file to save the data.
    """
    with open(filename, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }, f)

processed_data_path = config['data']['processed_data_path']
filename = processed_data_path + '/magnitude_cnn_data.pkl'
save_processed_data(X_train_all, X_val_final, X_test_final, y_train_all, y_val_final, y_test_final, 
                    filename)
print(f"Processed data saved to {filename}")