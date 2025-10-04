"""
SQLAlchemy Models for ExoHunter Database

This module defines the database models for storing predictions, datasets,
and user sessions in the ExoHunter system.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
from datetime import datetime
import json
import uuid

class Base(DeclarativeBase):
    pass


class UserSession(Base):
    """
    Model for tracking user sessions and API usage.
    """
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    ip_address = Column(String(45), nullable=True)  # Support IPv6
    user_agent = Column(String(512), nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    last_activity = Column(DateTime, nullable=False, default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationship to predictions
    predictions = relationship("Prediction", back_populates="session")
    datasets = relationship("Dataset", back_populates="session")
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, session_id='{self.session_id}', ip='{self.ip_address}')>"
    
    def to_dict(self):
        """Convert session to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'is_active': self.is_active
        }


class Dataset(Base):
    """
    Model for storing uploaded datasets and their metadata.
    """
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('user_sessions.id'), nullable=True)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)  # File size in bytes
    num_rows = Column(Integer, nullable=True)
    num_columns = Column(Integer, nullable=True)
    column_names = Column(JSON, nullable=True)  # Store as JSON array
    data_source = Column(String(50), nullable=True)  # 'upload', 'kepler', 'k2', 'tess', etc.
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Dataset statistics
    has_labels = Column(Boolean, default=False)
    label_distribution = Column(JSON, nullable=True)  # Count of each class
    feature_statistics = Column(JSON, nullable=True)  # Min, max, mean, std for numerical features
    
    # Relationship to session and predictions
    session = relationship("UserSession", back_populates="datasets")
    predictions = relationship("Prediction", back_populates="dataset")
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', rows={self.num_rows}, cols={self.num_columns})>"
    
    def to_dict(self):
        """Convert dataset to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'name': self.name,
            'filename': self.filename,
            'file_size': self.file_size,
            'num_rows': self.num_rows,
            'num_columns': self.num_columns,
            'column_names': self.column_names,
            'data_source': self.data_source,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_active': self.is_active,
            'has_labels': self.has_labels,
            'label_distribution': self.label_distribution,
            'feature_statistics': self.feature_statistics
        }


class Prediction(Base):
    """
    Model for storing prediction results and associated metadata.
    """
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('user_sessions.id'), nullable=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=True)
    
    # Prediction metadata
    prediction_type = Column(String(50), nullable=False)  # 'features', 'lightcurve', 'upload'
    model_type = Column(String(50), nullable=False)  # 'ensemble', 'xgboost', 'random_forest', 'mlp', 'mock'
    model_version = Column(String(100), nullable=True)
    
    # Input data
    input_features = Column(JSON, nullable=True)  # Store feature array as JSON
    input_metadata = Column(JSON, nullable=True)  # Additional input information
    feature_names = Column(JSON, nullable=True)  # Names of the features
    
    # Light curve specific data
    time_series = Column(JSON, nullable=True)  # Time array for light curves
    flux_series = Column(JSON, nullable=True)  # Flux array for light curves
    
    # Prediction results
    predicted_label = Column(String(50), nullable=False)
    predicted_probability = Column(Float, nullable=False)
    confidence_level = Column(String(20), nullable=False)  # 'HIGH', 'MEDIUM', 'LOW'
    all_probabilities = Column(JSON, nullable=False)  # All class probabilities as JSON
    
    # Performance metrics
    prediction_time_ms = Column(Float, nullable=True)  # Time taken for prediction in milliseconds
    preprocessing_time_ms = Column(Float, nullable=True)  # Time taken for preprocessing
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=func.now())
    
    # Quality indicators
    is_outlier = Column(Boolean, default=False)  # Flag for potential outliers
    uncertainty_score = Column(Float, nullable=True)  # Model uncertainty if available
    
    # Relationships
    session = relationship("UserSession", back_populates="predictions")
    dataset = relationship("Dataset", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, type='{self.prediction_type}', label='{self.predicted_label}', prob={self.predicted_probability:.3f})>"
    
    def to_dict(self):
        """Convert prediction to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'dataset_id': self.dataset_id,
            'prediction_type': self.prediction_type,
            'model_type': self.model_type,
            'model_version': self.model_version,
            'input_features': self.input_features,
            'input_metadata': self.input_metadata,
            'feature_names': self.feature_names,
            'time_series': self.time_series,
            'flux_series': self.flux_series,
            'predicted_label': self.predicted_label,
            'predicted_probability': self.predicted_probability,
            'confidence_level': self.confidence_level,
            'all_probabilities': self.all_probabilities,
            'prediction_time_ms': self.prediction_time_ms,
            'preprocessing_time_ms': self.preprocessing_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_outlier': self.is_outlier,
            'uncertainty_score': self.uncertainty_score
        }
    
    def get_confidence_score(self):
        """Calculate a confidence score based on probability distribution."""
        if not self.all_probabilities:
            return 0.0
        
        probs = list(self.all_probabilities.values()) if isinstance(self.all_probabilities, dict) else self.all_probabilities
        if len(probs) < 2:
            return 1.0
        
        # Sort probabilities in descending order
        sorted_probs = sorted(probs, reverse=True)
        # Confidence is the difference between top two probabilities
        return sorted_probs[0] - sorted_probs[1]
    
    def get_entropy(self):
        """Calculate entropy of the probability distribution."""
        import math
        
        if not self.all_probabilities:
            return float('inf')
        
        probs = list(self.all_probabilities.values()) if isinstance(self.all_probabilities, dict) else self.all_probabilities
        entropy = 0.0
        
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy


class APIRequest(Base):
    """
    Model for logging API requests for monitoring and analytics.
    """
    __tablename__ = 'api_requests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('user_sessions.id'), nullable=True)
    
    # Request details
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)  # GET, POST, PUT, DELETE
    status_code = Column(Integer, nullable=False)
    
    # Request metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)
    request_headers = Column(JSON, nullable=True)
    request_size = Column(Integer, nullable=True)  # Request body size in bytes
    response_size = Column(Integer, nullable=True)  # Response body size in bytes
    
    # Performance metrics
    response_time_ms = Column(Float, nullable=False)  # Response time in milliseconds
    cpu_usage_percent = Column(Float, nullable=True)  # CPU usage during request
    memory_usage_mb = Column(Float, nullable=True)  # Memory usage in MB
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    error_type = Column(String(100), nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False, default=func.now())
    
    # Relationship
    session = relationship("UserSession")
    
    def __repr__(self):
        return f"<APIRequest(id={self.id}, endpoint='{self.endpoint}', status={self.status_code}, time={self.response_time_ms}ms)>"
    
    def to_dict(self):
        """Convert API request to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'endpoint': self.endpoint,
            'method': self.method,
            'status_code': self.status_code,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_headers': self.request_headers,
            'request_size': self.request_size,
            'response_size': self.response_size,
            'response_time_ms': self.response_time_ms,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'stack_trace': self.stack_trace,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
