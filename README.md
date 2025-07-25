# Fall Detection Using Body Sensor Data

This project implements a machine learning solution for fall detection using data from body-worn sensors. The project was developed as part of the Prosigliere Coding Challenge and uses statistical features extracted from time-series sensor data to classify falls vs. non-fall activities.

## Project Overview

The main goal is to predict, based on body sensor data, whether a fall took place during a trial. The dataset consists of 480 trials from 8 subjects performing an average of 60 trials each.

### Data Description

- **Sensors**: 7 body sensors (right and left ankle, right and left thigh, head, sternum, and waist)
- **Features per sensor**: 9 features (3D acceleration, angular velocity, and magnetic field)
- **Trial types**: ADLs (Activities of Daily Living), Falls, and Near_Falls
- **Data format**: Excel files with time series data

## Project Structure

```
CodeChallengeFallDetection/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── cached/
├── models/                    # Saved trained models
│   ├── latest_models.yaml     # Latest model information
│   ├── {model_name}_{timestamp}.pkl
│   ├── scaler_{timestamp}.pkl
│   └── training_metadata_{timestamp}.yaml
├── plots/                     # Generated evaluation plots
│   ├── confusion_matrix_{model}_{timestamp}.png
│   ├── feature_importance_{model}_{timestamp}.png
│   └── evaluation_results_{timestamp}.yaml
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessor.py    # Data preprocessing
│   ├── features/
│   │   ├── engineering.py     # Feature engineering
│   │   └── statistical.py     # Statistical feature extraction
│   ├── models/
│   │   ├── trainer.py         # Model training
│   │   └── evaluator.py       # Model evaluation
│   └── utils/
│       └── helpers.py
├── notebooks/
│   └── fall_detection_analysis.ipynb
├── scripts/
│   ├── train_model.py         # Main training script
│   └── evaluate_model.py      # Evaluation script
└── tests/
    └── ...
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fall-detection.git
cd fall-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Place your data in the `data/raw/` directory following this structure:
```
data/raw/
├── sub1/
│   ├── ADLs/
│   ├── Falls/
│   └── Near_Falls/
├── sub2/
│   ├── ADLs/
│   ├── Falls/
│   └── Near_Falls/
└── ...
```

### 2. Configuration

Modify `config/config.yaml` to adjust parameters:
- Data paths
- Sensor locations and features
- Model hyperparameters
- Train/validation/test split ratios

### 3. Training Models

Run the main training script:
```bash
python scripts/train_model.py
```

This will:
- Load and preprocess the data
- Extract statistical features
- Split data by subjects (to avoid data leakage)
- Train multiple models (Random Forest, Logistic Regression)
- **Save trained models** with timestamps in the `models/` directory
- **Save the fitted scaler** for consistent preprocessing
- **Save training metadata** for reproducibility
- Evaluate and display results

#### Model Saving Details

The training script automatically saves:
- **Individual models**: `{model_name}_{timestamp}.pkl`
- **Scaler**: `scaler_{timestamp}.pkl` (for consistent feature normalization)
- **Training metadata**: `training_metadata_{timestamp}.yaml` (configuration and training info)
- **Latest models info**: `latest_models.yaml` (points to most recent training run)

### 4. Evaluating Models

Evaluate all trained models:
```bash
python scripts/evaluate_model.py
```

Evaluate a specific model:
```bash
python scripts/evaluate_model.py random_forest
```

The evaluation script will:
- **Load the latest trained models** automatically
- **Use the same preprocessing** (scaler and data splits) as training
- Generate detailed classification reports
- **Create visualizations**:
  - Confusion matrices
  - Feature importance plots (for both tree-based and linear models)
- **Save timestamped results** in the `plots/` directory
- Save evaluation summaries as YAML files

### 5. Loading Saved Models

To load and use trained models in your own scripts:

```python
import pickle
import yaml

# Load latest model information
with open('models/latest_models.yaml', 'r') as f:
    latest_info = yaml.safe_load(f)

# Load a specific model
with open(f'models/{latest_info["models"]["logistic_regression"]}', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open(f'models/{latest_info["scaler"]}', 'rb') as f:
    scaler = pickle.load(f)

# Use for prediction
X_new_normalized = scaler.transform(X_new)
predictions = model.predict(X_new_normalized)
```

## Features

### Feature Engineering
- **Magnitude calculation**: For each sensor, computes the magnitude of 3D vectors (acceleration, angular velocity, magnetic field)
- **Statistical features**: Extracts 12 statistical measures per feature:
  - Mean, standard deviation, min, max, median
  - 25th and 75th percentiles, variance
  - Mean absolute difference, standard deviation of differences
  - Zero crossings, outlier count

### Models Implemented
1. **Random Forest**: Ensemble method with balanced class weights
2. **Logistic Regression**: Linear classifier with L2 regularization
3. **Logistic Regression L1**: Feature selection through L1 penalty
4. **CNN**: Running outside of the main train/eval pipeline for now, use:

```python
python src/data/prep_cnn_data.py
python scripts/train_cnn_model.py

### Visualization Features
- **Confusion matrices**: Visual representation of model performance
- **Feature importance plots**: 
  - Tree-based models: Built-in feature importances
  - Linear models: Absolute coefficients as importance measures
- **Timestamped outputs**: All plots and results include training timestamps for organization

## Reproducibility

The project ensures full reproducibility through:
- **Saved training configurations**: All hyperparameters and settings are preserved
- **Consistent data splits**: Uses the same random seed and split parameters
- **Saved preprocessing**: The fitted scaler ensures identical feature normalization
- **Timestamped artifacts**: All models and results are timestamped for version tracking

## File Organization

### Models Directory (`models/`)
- `latest_models.yaml`: Information about the most recent training run
- `{model_name}_{timestamp}.pkl`: Individual trained models
- `scaler_{timestamp}.pkl`: Fitted StandardScaler for preprocessing
- `training_metadata_{timestamp}.yaml`: Complete training configuration and metadata

### Plots Directory (`plots/`)
- `confusion_matrix_{model}_{timestamp}.png`: Model confusion matrices
- `feature_importance_{model}_{timestamp}.png`: Feature importance visualizations
- `feature_importance_{model}_{timestamp}.csv`: Feature importance data
- `evaluation_results_{timestamp}.yaml`: Comprehensive evaluation summaries