"""
Advanced deep learning models for exoplanet classification.

This module provides neural network implementations including:
1. Multi-layer perceptron (MLP) for tabular features with dropout and early stopping
2. 1D Convolutional Neural Network (CNN) for raw light curve time series data

Both models include proper regularization, early stopping, and model persistence.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")
    # Create dummy classes for type hints
    class keras:
        class Model: pass
        class callbacks:
            class History: pass


class TabularMLP:
    """
    Multi-layer perceptron for tabular exoplanet feature classification.
    
    This model is designed for processed tabular features like orbital period,
    planet radius, stellar parameters, etc. It includes dropout for regularization
    and early stopping to prevent overfitting.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        output_activation: str = 'softmax'
    ):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network models")
            
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_activation = output_activation
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """Build the MLP model architecture."""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.BatchNormalization()
        ])
        
        # Add hidden layers with dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units, 
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(0.001),
                name=f'hidden_{i+1}'
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(
            self.num_classes,
            activation=self.output_activation,
            name='output'
        ))
        
        return model
    
    def compile_model(
        self, 
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'sparse_categorical_crossentropy',
        metrics: List[str] = ['accuracy']
    ):
        """Compile the model with specified optimizer and loss function."""
        if self.model is None:
            self.model = self.build_model()
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
            
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
    def prepare_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and scale the input data."""
        # Initialize scalers if not already done
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)
            
        return X_scaled, y_encoded
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the MLP model with early stopping.
        
        Args:
            X: Input features DataFrame
            y: Target labels Series
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Training history object
        """
        # Prepare data
        X_scaled, y_encoded = self.prepare_data(X, y)
        
        # Build and compile model if not done
        if self.model is None:
            self.compile_model()
        
        # Define callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=verbose
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=verbose
        )
        
        # Train the model
        self.history = self.model.fit(
            X_scaled, y_encoded,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Only scale features, don't transform labels for prediction
        if self.scaler is None:
            raise ValueError("Model scaler not fitted")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }


def make_lightcurve_cnn(
    input_length: int,
    num_classes: int = 3,
    num_filters: List[int] = [32, 64, 128],
    kernel_sizes: List[int] = [7, 5, 3],
    pool_sizes: List[int] = [2, 2, 2],
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Create a 1D CNN model for light curve classification.
    
    This model is designed for raw light curve time series data, using 1D
    convolutions to detect transit patterns and other temporal features.
    
    Args:
        input_length: Length of input light curve sequences
        num_classes: Number of output classes (confirmed, candidate, false_positive)
        num_filters: Number of filters for each conv layer
        kernel_sizes: Kernel sizes for each conv layer
        pool_sizes: Max pooling sizes for each layer
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
        
    Example:
        >>> model = make_lightcurve_cnn(input_length=1000, num_classes=3)
        >>> model.summary()
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for neural network models")
    
    # Input layer
    inputs = layers.Input(shape=(input_length, 1), name='lightcurve_input')
    x = inputs
    
    # Add normalization
    x = layers.BatchNormalization(name='input_batch_norm')(x)
    
    # Convolutional blocks
    for i, (filters, kernel_size, pool_size) in enumerate(zip(num_filters, kernel_sizes, pool_sizes)):
        # 1D Convolution
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name=f'conv1d_{i+1}'
        )(x)
        
        # Batch normalization
        x = layers.BatchNormalization(name=f'batch_norm_conv_{i+1}')(x)
        
        # Max pooling
        x = layers.MaxPooling1D(
            pool_size=pool_size,
            name=f'maxpool_{i+1}'
        )(x)
        
        # Dropout
        x = layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}')(x)
    
    # Global average pooling to reduce parameters
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='batch_norm_dense_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_dense_1')(x)
    
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='batch_norm_dense_2')(x)
    x = layers.Dropout(dropout_rate, name='dropout_dense_2')(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='classification_output'
    )(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='lightcurve_cnn')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_tabular(
    X: pd.DataFrame,
    y: pd.Series,
    model_save_path: Union[str, Path],
    test_size: float = 0.2,
    val_size: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 42,
    **model_kwargs
) -> Tuple[TabularMLP, Dict[str, float]]:
    """
    Train a tabular MLP model for exoplanet classification.
    
    Args:
        X: Input features DataFrame
        y: Target labels Series
        model_save_path: Path to save the trained model
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        epochs: Maximum training epochs
        batch_size: Training batch size
        random_state: Random seed for reproducibility
        **model_kwargs: Additional arguments for TabularMLP
        
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    print("Training tabular MLP model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Get number of classes and features
    num_classes = len(y.unique())
    input_dim = X.shape[1]
    
    print(f"Dataset info: {len(X)} samples, {input_dim} features, {num_classes} classes")
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    
    # Initialize and train model
    model = TabularMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        **model_kwargs
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=val_size,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    save_model(model, model_save_path, model_type='tabular')
    print(f"Model saved to {model_save_path}")
    
    return model, test_metrics


def train_lightcurve(
    X: np.ndarray,
    y: np.ndarray,
    model_save_path: Union[str, Path],
    test_size: float = 0.2,
    val_size: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 42,
    **model_kwargs
) -> Tuple[keras.Model, Dict[str, float]]:
    """
    Train a 1D CNN model for light curve classification.
    
    Args:
        X: Input light curve arrays (samples, time_steps)
        y: Target labels array
        model_save_path: Path to save the trained model
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        epochs: Maximum training epochs
        batch_size: Training batch size
        random_state: Random seed for reproducibility
        **model_kwargs: Additional arguments for make_lightcurve_cnn
        
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    print("Training light curve CNN model...")
    
    # Prepare data
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Add channel dimension
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Get model parameters
    input_length = X.shape[1]
    num_classes = len(np.unique(y))
    
    print(f"Dataset info: {len(X)} samples, {input_length} time steps, {num_classes} classes")
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    
    # Create model
    model = make_lightcurve_cnn(
        input_length=input_length,
        num_classes=num_classes,
        **model_kwargs
    )
    
    print(f"Model architecture:")
    model.summary()
    
    # Define callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=val_size,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Generate predictions for detailed metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate additional metrics
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    
    test_metrics = {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }
    
    print(f"\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    save_keras_model(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model, test_metrics


def save_model(
    model: TabularMLP,
    save_path: Union[str, Path],
    model_type: str = 'tabular'
) -> None:
    """
    Save a trained model with all necessary components.
    
    Args:
        model: Trained model object
        save_path: Path to save the model
        model_type: Type of model ('tabular' or 'lightcurve')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'tabular':
        # Save the Keras model
        model.model.save(str(save_path / 'model.h5'))
        
        # Save preprocessing objects
        import joblib
        joblib.dump(model.scaler, save_path / 'scaler.joblib')
        joblib.dump(model.label_encoder, save_path / 'label_encoder.joblib')
        
        # Save model configuration
        config = {
            'input_dim': model.input_dim,
            'num_classes': model.num_classes,
            'hidden_layers': model.hidden_layers,
            'dropout_rate': model.dropout_rate,
            'activation': model.activation,
            'output_activation': model.output_activation,
            'model_type': model_type
        }
        
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Model saved to {save_path}")


def save_keras_model(model: keras.Model, save_path: Union[str, Path]) -> None:
    """Save a Keras model with architecture and weights."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save complete model
    model.save(str(save_path / 'model.h5'))
    
    # Save architecture separately
    with open(save_path / 'architecture.json', 'w') as f:
        f.write(model.to_json())
    
    # Save weights separately (using correct filename)
    model.save_weights(str(save_path / 'model.weights.h5'))
    
    print(f"Keras model saved to {save_path}")


def load_model(load_path: Union[str, Path], model_type: str = 'tabular') -> Union[TabularMLP, keras.Model]:
    """
    Load a saved model.
    
    Args:
        load_path: Path to the saved model
        model_type: Type of model ('tabular' or 'lightcurve')
        
    Returns:
        Loaded model object
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required to load neural network models")
    
    load_path = Path(load_path)
    
    if model_type == 'tabular':
        # Load configuration
        with open(load_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = TabularMLP(**{k: v for k, v in config.items() if k != 'model_type'})
        
        # Load Keras model
        model.model = keras.models.load_model(str(load_path / 'model.h5'))
        
        # Load preprocessing objects
        import joblib
        model.scaler = joblib.load(load_path / 'scaler.joblib')
        model.label_encoder = joblib.load(load_path / 'label_encoder.joblib')
        
        return model
    
    elif model_type == 'lightcurve':
        # Load Keras model
        return keras.models.load_model(str(load_path / 'model.h5'))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage and testing functions
def generate_sample_tabular_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate sample tabular data for testing."""
    np.random.seed(42)
    
    # Generate realistic exoplanet features
    data = {
        'orbital_period': np.random.lognormal(2, 1.5, n_samples),
        'planet_radius': np.random.lognormal(0.5, 0.8, n_samples),
        'stellar_mass': np.random.normal(1.0, 0.3, n_samples),
        'stellar_radius': np.random.normal(1.0, 0.4, n_samples),
        'stellar_temp': np.random.normal(5800, 800, n_samples),
        'transit_depth': np.random.exponential(0.001, n_samples),
        'transit_duration': np.random.exponential(5, n_samples),
        'signal_to_noise': np.random.exponential(10, n_samples)
    }
    
    X = pd.DataFrame(data)
    
    # Generate labels based on features (simulate realistic distributions)
    probabilities = np.zeros((n_samples, 3))
    
    # Confirmed planets: larger signals, reasonable periods
    confirmed_mask = (X['signal_to_noise'] > 15) & (X['orbital_period'] < 100)
    probabilities[confirmed_mask, 0] = 0.7
    
    # False positives: often have extreme values
    fp_mask = (X['signal_to_noise'] < 5) | (X['transit_depth'] > 0.01)
    probabilities[fp_mask, 2] = 0.6
    
    # Candidates: moderate signals
    probabilities[:, 1] = 0.4  # Base candidate probability
    
    # Normalize probabilities
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Sample labels
    labels = np.array([np.random.choice(3, p=prob) for prob in probabilities])
    label_names = ['confirmed', 'candidate', 'false_positive']
    y = pd.Series([label_names[label] for label in labels])
    
    return X, y


def generate_sample_lightcurve_data(
    n_samples: int = 1000, 
    sequence_length: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample light curve data for testing."""
    np.random.seed(42)
    
    X = np.zeros((n_samples, sequence_length))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Base noise
        noise = np.random.normal(0, 0.001, sequence_length)
        
        if i < n_samples // 3:  # Confirmed planets
            # Add transit signal
            transit_depth = np.random.uniform(0.001, 0.01)
            transit_duration = np.random.randint(5, 20)
            transit_start = np.random.randint(0, sequence_length - transit_duration)
            
            signal = np.ones(sequence_length)
            signal[transit_start:transit_start + transit_duration] -= transit_depth
            X[i] = signal + noise
            y[i] = 0  # confirmed
            
        elif i < 2 * n_samples // 3:  # Candidates
            # Weaker signal or partial transit
            if np.random.random() < 0.5:
                transit_depth = np.random.uniform(0.0005, 0.003)
                transit_duration = np.random.randint(3, 15)
                transit_start = np.random.randint(0, sequence_length - transit_duration)
                
                signal = np.ones(sequence_length)
                signal[transit_start:transit_start + transit_duration] -= transit_depth
                X[i] = signal + noise
            else:
                X[i] = np.ones(sequence_length) + noise
            y[i] = 1  # candidate
            
        else:  # False positives
            # Random artifacts or stellar variability
            if np.random.random() < 0.3:
                # Stellar variability
                variability = 0.002 * np.sin(2 * np.pi * np.arange(sequence_length) / 100)
                X[i] = np.ones(sequence_length) + variability + noise
            else:
                # Just noise
                X[i] = np.ones(sequence_length) + noise
            y[i] = 2  # false_positive
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Generating sample data...")
    
    # Test tabular MLP
    X_tab, y_tab = generate_sample_tabular_data(1000)
    print(f"Tabular data shape: {X_tab.shape}, labels: {y_tab.value_counts().to_dict()}")
    
    # Test light curve CNN data
    X_lc, y_lc = generate_sample_lightcurve_data(500, 1000)
    print(f"Light curve data shape: {X_lc.shape}, labels: {np.bincount(y_lc)}")
    
    if TENSORFLOW_AVAILABLE:
        print("\nTensorFlow is available. Models can be trained.")
        
        # Test model creation
        try:
            mlp = TabularMLP(input_dim=8, num_classes=3)
            print("✓ TabularMLP created successfully")
            
            cnn = make_lightcurve_cnn(input_length=1000, num_classes=3)
            print("✓ Light curve CNN created successfully")
            print(f"CNN has {cnn.count_params():,} parameters")
            
        except Exception as e:
            print(f"Error creating models: {e}")
    else:
        print("\nTensorFlow not available. Install with: pip install tensorflow")
