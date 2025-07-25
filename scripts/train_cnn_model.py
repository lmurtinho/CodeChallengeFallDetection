### CNN model

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For GPU determinism (if using GPU)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Configure TensorFlow for deterministic behavior
    tf.config.experimental.enable_op_determinism()

# Set seeds at the beginning
set_random_seeds(42)

class FallDetectionCNN_MagnitudeOnly_FullSequence:
    """CNN for fall detection using magnitude features only with full sequence length."""
    
    def __init__(self, sequence_length: int, n_features: int, n_classes: int = 2, random_seed: int = 42):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.model = None
        self.history = None
        
        # Set seeds for this instance
        set_random_seeds(self.random_seed)
        
    def build_model(self):
        """Build the CNN architecture."""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # First Conv1D block
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        # Second Conv1D block
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        # Third Conv1D block
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='FallDetectionCNN_Magnitude_FullSeq')
        
        # Compile model with reproducible optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # Set optimizer seed for reproducibility
        optimizer.build(self.model.trainable_variables)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 50, batch_size: int = 32, verbose: int = 1):
        """Train the model."""
        if self.model is None:
            self.build_model()
            
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        probabilities = self.model.predict(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on test data."""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted')
        }
        
        # Add ROC AUC if binary classification
        if self.n_classes == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, probabilities[:, 1])
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        # Use native Keras format (.keras) instead of legacy HDF5 (.h5)
        if not filepath.endswith('.keras'):
            filepath = filepath.replace('.h5', '.keras')
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)

# retrieve the data from the pickled file
import pickle
def load_processed_data(filename: str):
    """
    Load processed data from a pickle file.
    
    Args:
        filename: Name of the file to load the data from.
        
    Returns:
        Dictionary containing training, validation, and test data.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# get filename from config
import yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
filename = config['data']['processed_data_path'] + '/magnitude_cnn_data.pkl'
data = load_processed_data(filename)

X_train_all = data['X_train']
y_train_all = data['y_train']
X_val_final = data['X_val']
y_val_final = data['y_val']
X_test_final = data['X_test']
y_test_final = data['y_test']

print(f"Loaded data shapes: X_train={X_train_all.shape}, y_train={y_train_all.shape}, "
      f"X_val={X_val_final.shape}, y_val={y_val_final.shape}, "
      f"X_test={X_test_final.shape}, y_test={y_test_final.shape}")

# Initialize model with fixed random seed
model = FallDetectionCNN_MagnitudeOnly_FullSequence(
    sequence_length=2562,
    n_features=21,
    n_classes=2,
    random_seed=42
)
    
# Build model to see architecture
keras_model = model.build_model()
print("\nFull-Sequence Magnitude-Only CNN Architecture:")
keras_model.summary()
    
# Count parameters
total_params = keras_model.count_params()
print(f"\nTotal parameters: {total_params:,}")

print(f"Final training: {len(X_train_all)} trials")
print(f"Final validation: {len(X_val_final)} trials")

final_model = FallDetectionCNN_MagnitudeOnly_FullSequence(
    sequence_length=2562,
    n_features=21,
    n_classes=2,
    random_seed=42
)

final_model.fit(
    # X_train_final, y_train_final,
    X_train_all, y_train_all,
    X_val_final, y_val_final,
    epochs=50, batch_size=32, verbose=1
)

# Evaluate on final test set
test_metrics = final_model.evaluate(X_test_final, y_test_final)
test_predictions = final_model.predict(X_test_final)

print(f"\nTest Set Results (n={len(y_test_final)} trials):")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

# show confusion matrix for test set
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)
# get current date and time for saving plots
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("\nConfusion Matrix for Test Set:")
cm = confusion_matrix(y_test_final, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fall'], yticklabels=['Normal', 'Fall'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Test Set')
# save the confusion matrix plot
plt.savefig(f'plots/confusion_matrix_test_set_{timestamp}.png')

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

# Get prediction probabilities for the positive class (Fall) on the validation set
y_pred_proba = final_model.predict_proba(X_val_final)[:, 1]

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_val_final, y_pred_proba)

# Calculate F1 score for each threshold
f1_scores = np.array([f1_score(y_val_final, (y_pred_proba >= t)) for t in thresholds])

# Find the threshold that maximizes the F1 score
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]
best_f1_score = f1_scores[best_threshold_index]

print(f"Best Threshold: {best_threshold:.4f}")
print(f"F1 Score at Best Threshold: {best_f1_score:.4f}")

# Plot Precision-Recall curve and indicate the best threshold
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.scatter(recall[best_threshold_index], precision[best_threshold_index], color='red', s=100, label=f'Best Threshold ({best_threshold:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Best Threshold')
plt.legend()
plt.grid()
# save the precision-recall curve plot
plt.savefig(f'plots/precision_recall_curve_best_threshold_{timestamp}.png')

# Evaluate on test set using the best threshold
test_predictions_thresholded = (final_model.predict_proba(X_test_final)[:, 1] >= best_threshold).astype(int)

print(f"\nFinal Test Set Results with Best Threshold (n={len(y_test_final)} trials):")

# Calculate and print metrics
metrics_thresholded = {
    'accuracy': accuracy_score(y_test_final, test_predictions_thresholded),
    'precision': precision_score(y_test_final, test_predictions_thresholded),
    'recall': recall_score(y_test_final, test_predictions_thresholded),
    'f1_score': f1_score(y_test_final, test_predictions_thresholded)
}

for metric, value in metrics_thresholded.items():
    print(f"  {metric}: {value:.4f}")

# Show confusion matrix for test set with thresholded predictions
cm_thresholded = confusion_matrix(y_test_final, test_predictions_thresholded)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_thresholded, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fall'], yticklabels=['Normal', 'Fall'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Test Set (Thresholded)')
# save the confusion matrix plot
plt.savefig(f'plots/confusion_matrix_test_set_thresholded_{timestamp}.png')

# save the final model
    # Create timestamp for this training run
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'fall_detection_cnn_magnitude_full_sequence_{timestamp}.keras'
final_model.save_model(f'models/{filename}')