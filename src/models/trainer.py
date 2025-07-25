"""
Model training utilities for fall detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List
from .cnn_model import FallDetectionCNN, prepare_sequence_data


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


def train_cnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Train a CNN model on sequence data.
    
    Args:
        X_train: Training sequences (samples, timesteps, features)
        y_train: Training labels
        X_test: Test sequences
        y_test: Test labels
        X_val: Validation sequences (optional)
        y_val: Validation labels (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary containing model and evaluation metrics
    """
    print(f"\nTraining CNN model...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize CNN model
    cnn = FallDetectionCNN(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        n_classes=2
    )
    
    # Train model
    history = cnn.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate model
    metrics = cnn.evaluate(X_test, y_test)
    
    # Add model and training history to results
    result = {
        'model': cnn,
        'history': history,
        'model_name': 'CNN',
        **metrics
    }
    
    print(f"CNN Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"CNN Test F1-Score: {metrics['f1_score']:.4f}")
    
    return result


def train_hybrid_approach(
    stat_X_train: pd.DataFrame,
    stat_y_train: pd.Series,
    stat_X_test: pd.DataFrame,
    stat_y_test: pd.Series,
    seq_X_train: np.ndarray,
    seq_y_train: np.ndarray,
    seq_X_test: np.ndarray,
    seq_y_test: np.ndarray,
    seq_X_val: np.ndarray = None,
    seq_y_val: np.ndarray = None
) -> Dict[str, Dict[str, Any]]:
    """
    Train both statistical and CNN models.
    
    Args:
        stat_X_train: Statistical features training data
        stat_y_train: Statistical features training labels
        stat_X_test: Statistical features test data
        stat_y_test: Statistical features test labels
        seq_X_train: Sequence training data
        seq_y_train: Sequence training labels
        seq_X_test: Sequence test data
        seq_y_test: Sequence test labels
        seq_X_val: Sequence validation data (optional)
        seq_y_val: Sequence validation labels (optional)
        
    Returns:
        Dictionary containing results for all models
    """
    results = {}
    
    # Train statistical models
    print("Training statistical models...")
    stat_results = train_statistical_approach(
        stat_X_train, stat_y_train, stat_X_test, stat_y_test
    )
    results.update(stat_results)
    
    # Train CNN model
    print("Training CNN model...")
    cnn_result = train_cnn_model(
        seq_X_train, seq_y_train,
        seq_X_test, seq_y_test,
        seq_X_val, seq_y_val
    )
    results['CNN'] = cnn_result
    
    return results


def subject_leave_one_out_cv(
    X: pd.DataFrame,
    y: pd.Series,
    subs: pd.Series,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    all_trials: List[Dict] = None,
    include_cnn: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform leave-one-out cross-validation at subject level using train and validation sets.
    
    Args:
        X: Feature matrix
        y: Labels
        subs: Subject IDs for each sample
        train_indices: Indices of training samples
        val_indices: Indices of validation samples
        all_trials: List of trial dictionaries (required for CNN)
        include_cnn: Whether to include CNN model in cross-validation
        
    Returns:
        Dictionary with results for each model across all CV folds
    """
    from models.evaluator import evaluate_model
    from data.preprocessor import normalize_features
    
    # Combine train and validation sets for LOOCV
    cv_indices = np.concatenate([train_indices, val_indices])
    X_cv = X.loc[cv_indices]
    y_cv = y.loc[cv_indices]
    subs_cv = subs.loc[cv_indices]
    
    # Get unique subjects in the CV set
    unique_subjects = sorted(subs_cv.unique())
    
    print(f"Performing leave-one-out CV with {len(unique_subjects)} subjects...")
    
    # Initialize results dictionary
    model_configs = get_model_configs()
    results = {model_name: [] for model_name in model_configs.keys()}
    
    # Add CNN to results if requested
    if include_cnn:
        results['CNN'] = []
        
    # Prepare sequence data for CNN if needed
    seq_X_cv = None
    seq_y_cv = None
    seq_subs_cv = None
    
    if include_cnn and all_trials is not None:
        # Filter trials for CV subjects
        cv_trials = []
        for trial in all_trials:
            if trial['subject_id'] in subs_cv.values:
                cv_trials.append(trial)
        
        # Prepare sequence data
        seq_X_cv, seq_y_cv, seq_subs_cv = prepare_sequence_data(cv_trials)
        print(f"Prepared {len(seq_X_cv)} sequence windows for CNN CV")
    
    # Perform leave-one-out CV
    for i, test_subject in enumerate(unique_subjects):
        print(f"Fold {i+1}/{len(unique_subjects)}: Testing on subject {test_subject}")
        
        # Split data: one subject for test, rest for train
        test_mask = subs_cv == test_subject
        train_mask = ~test_mask
        
        X_train_fold = X_cv[train_mask]
        y_train_fold = y_cv[train_mask]
        X_test_fold = X_cv[test_mask]
        y_test_fold = y_cv[test_mask]
        
        # Normalize features for this fold
        scaler = StandardScaler()
        X_train_norm, _, X_test_norm = normalize_features(
            X_train_fold, X_train_fold, X_test_fold, scaler
        )
        
        # Convert back to DataFrame for consistency
        X_train_norm = pd.DataFrame(X_train_norm, columns=X_cv.columns)
        X_test_norm = pd.DataFrame(X_test_norm, columns=X_cv.columns)
        
        # Train and evaluate each model
        for model_name, model in model_configs.items():
            try:
                # Train model
                model.fit(X_train_norm, y_train_fold)
                
                # Evaluate model
                fold_result = evaluate_model(model, X_test_norm, y_test_fold, f"{model_name}_fold_{i+1}")
                fold_result['test_subject'] = test_subject
                fold_result['fold'] = i + 1
                
                results[model_name].append(fold_result)
                
            except Exception as e:
                print(f"Error training {model_name} on fold {i+1}: {e}")
                continue
        
        # Train CNN model if requested
        if include_cnn and seq_X_cv is not None:
            try:
                # Split sequence data for this fold
                seq_test_mask = seq_subs_cv == test_subject
                seq_train_mask = ~seq_test_mask
                
                seq_X_train_fold = seq_X_cv[seq_train_mask]
                seq_y_train_fold = seq_y_cv[seq_train_mask]
                seq_X_test_fold = seq_X_cv[seq_test_mask]
                seq_y_test_fold = seq_y_cv[seq_test_mask]
                
                if len(seq_X_train_fold) > 0 and len(seq_X_test_fold) > 0:
                    # Train CNN model
                    cnn_result = train_cnn_model(
                        seq_X_train_fold, seq_y_train_fold,
                        seq_X_test_fold, seq_y_test_fold,
                        epochs=50,  # Reduced epochs for CV
                        batch_size=16
                    )
                    
                    cnn_result['test_subject'] = test_subject
                    cnn_result['fold'] = i + 1
                    results['CNN'].append(cnn_result)
                    
            except Exception as e:
                print(f"Error training CNN on fold {i+1}: {e}")
                continue
    
    # Calculate average performance across folds
    print("\nCalculating average performance across folds...")
    for model_name in results.keys():
        if results[model_name]:
            fold_results = results[model_name]
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
            avg_metrics = {}
            for metric in metrics:
                values = [fold[metric] for fold in fold_results if metric in fold]
                if values:
                    avg_metrics[f'avg_{metric}'] = np.mean(values)
                    avg_metrics[f'std_{metric}'] = np.std(values)
            
            # Add summary to results
            summary = {
                'model_name': f"{model_name}_LOOCV_Summary",
                'n_folds': len(fold_results),
                'test_subjects': [fold['test_subject'] for fold in fold_results],
                **avg_metrics
            }
            results[model_name].append(summary)
    
    return results


def train_cnn_model(
    seq_X_train: np.ndarray,
    seq_y_train: np.ndarray,
    seq_X_test: np.ndarray,
    seq_y_test: np.ndarray,
    seq_X_val: np.ndarray = None,
    seq_y_val: np.ndarray = None,
    all_trials: List[Dict] = None,
    test_subjects: set = None,
    config: Dict = None,
    epochs: int = 100,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Train CNN model with both sequence-level and trial-level validation.
    
    Args:
        seq_X_train: Training sequences
        seq_y_train: Training labels
        seq_X_test: Test sequences
        seq_y_test: Test labels
        seq_X_val: Validation sequences (optional)
        seq_y_val: Validation labels (optional)
        all_trials: All trial data for trial-level validation (optional)
        test_subjects: Set of test subjects for trial-level validation (optional)
        config: Configuration dictionary (optional)
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Dictionary containing both sequence-level and trial-level metrics
    """
    
    print(f"Training CNN with {len(seq_X_train)} sequences")
    print(f"Sequence shape: {seq_X_train.shape}")
    
    # Initialize CNN model
    cnn = FallDetectionCNN(
        sequence_length=seq_X_train.shape[1],
        n_features=seq_X_train.shape[2],
        n_classes=2
    )
    
    # Build model
    cnn.build_model()
    
    # Train model
    print("Training CNN on sequence data...")
    history = cnn.fit(
        seq_X_train, seq_y_train,
        seq_X_val, seq_y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate on sequence-level test data
    print("Evaluating on sequence-level test data...")
    sequence_metrics = cnn.evaluate(seq_X_test, seq_y_test)
    
    # Initialize result dictionary
    result = {
        'model': cnn,
        'history': history,
        'sequence_level_metrics': sequence_metrics,
        'accuracy': sequence_metrics['accuracy'],  # Keep for backward compatibility
        'precision': sequence_metrics['precision'],
        'recall': sequence_metrics['recall'],
        'f1_score': sequence_metrics['f1_score']
    }
    
    if 'roc_auc' in sequence_metrics:
        result['roc_auc'] = sequence_metrics['roc_auc']
    
    # Add trial-level validation if trial data is provided
    if all_trials is not None and test_subjects is not None and config is not None:
        print("Performing trial-level validation...")
        
        try:
            # Import here to avoid circular imports
            from scripts.evaluate_cnn import evaluate_trial_level_performance
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Perform trial-level evaluation
                trial_result = evaluate_trial_level_performance(
                    cnn, all_trials, test_subjects, config, tmpdir
                )
                
                if trial_result[0] is not None:  # trial_df
                    trial_df, best_results, best_method = trial_result
                    
                    # Add trial-level metrics to result
                    result['trial_level_metrics'] = {
                        'trial_accuracy': best_results['accuracy'],
                        'trial_precision': best_results['precision'],
                        'trial_recall': best_results['recall'],
                        'trial_f1_score': best_results['f1_score'],
                        'best_aggregation_method': best_method,
                        'n_trials_evaluated': len(trial_df)
                    }
                    
                    print(f"Trial-level validation completed:")
                    print(f"  Method: {best_method}")
                    print(f"  Accuracy: {best_results['accuracy']:.4f}")
                    print(f"  F1-Score: {best_results['f1_score']:.4f}")
                    print(f"  Trials evaluated: {len(trial_df)}")
                else:
                    print("Warning: Trial-level validation failed")
                    
        except Exception as e:
            print(f"Warning: Could not perform trial-level validation: {e}")
            result['trial_level_metrics'] = None
    else:
        # No trial-level validation data provided
        result['trial_level_metrics'] = None
        if all_trials is None:
            print("Note: No trial data provided for trial-level validation")
    
    print(f"CNN training completed:")
    print(f"  Sequence-level Accuracy: {sequence_metrics['accuracy']:.4f}")
    print(f"  Sequence-level F1-Score: {sequence_metrics['f1_score']:.4f}")
    
    return result