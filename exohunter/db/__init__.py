"""
Database initialization and utility functions for ExoHunter.

This module provides database connection management, initialization helpers,
and functions for persisting predictions and logging API requests.
"""

import os
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import time
import psutil
import traceback

try:
    from sqlalchemy import create_engine, MetaData
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import StaticPool
    from .models import Base, Prediction, Dataset, UserSession, APIRequest
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    # Fallback to basic logging if SQLAlchemy is not available
    SQLALCHEMY_AVAILABLE = False
    print("Warning: SQLAlchemy not available. Database features will be limited.")

from logging.handlers import RotatingFileHandler


class DatabaseManager:
    """
    Manages database connections and provides utility functions for ExoHunter.
    """
    
    def __init__(self, database_url: str = None, log_dir: str = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection string. Defaults to SQLite in project root.
            log_dir: Directory for log files. Defaults to 'logs' in project root.
        """
        self.database_url = database_url or self._get_default_database_url()
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent.parent.parent / "logs"
        self.engine = None
        self.SessionLocal = None
        self._logger = None
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize database and logging
        self.init_database()
        self.setup_logging()
    
    def _get_default_database_url(self) -> str:
        """Get default SQLite database URL."""
        db_path = Path(__file__).parent.parent.parent / "exohunter.db"
        return f"sqlite:///{db_path}"
    
    def init_database(self):
        """Initialize database connection and create tables."""
        if not SQLALCHEMY_AVAILABLE:
            self._init_sqlite_fallback()
            return
        
        try:
            # Create SQLAlchemy engine
            if self.database_url.startswith("sqlite"):
                # SQLite specific configuration
                self.engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 20
                    },
                    echo=False
                )
            else:
                # Other databases (PostgreSQL, MySQL, etc.)
                self.engine = create_engine(
                    self.database_url,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    echo=False
                )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            print(f"Database initialized successfully: {self.database_url}")
            
        except Exception as e:
            print(f"Failed to initialize database: {e}")
            self._init_sqlite_fallback()
    
    def _init_sqlite_fallback(self):
        """Initialize basic SQLite connection as fallback."""
        try:
            db_path = Path(__file__).parent.parent.parent / "exohunter_fallback.db"
            self.fallback_db_path = str(db_path)
            
            # Create basic tables for logging
            conn = sqlite3.connect(self.fallback_db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prediction_type TEXT,
                    model_type TEXT,
                    predicted_label TEXT,
                    predicted_probability REAL,
                    all_probabilities TEXT,
                    input_data TEXT,
                    session_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT,
                    method TEXT,
                    status_code INTEGER,
                    response_time_ms REAL,
                    ip_address TEXT,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            print(f"Fallback SQLite database initialized: {self.fallback_db_path}")
            
        except Exception as e:
            print(f"Failed to initialize fallback database: {e}")
    
    def setup_logging(self):
        """Setup rotating file logger for API requests and predictions."""
        # Create logger
        self._logger = logging.getLogger('exohunter')
        self._logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)
        
        # API request logger
        api_log_file = self.log_dir / "api_requests.log"
        api_handler = RotatingFileHandler(
            api_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        api_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        api_handler.setFormatter(api_formatter)
        
        # Prediction logger
        prediction_log_file = self.log_dir / "predictions.log"
        prediction_handler = RotatingFileHandler(
            prediction_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        prediction_formatter = logging.Formatter(
            '%(asctime)s - PREDICTION - %(message)s'
        )
        prediction_handler.setFormatter(prediction_formatter)
        
        # Add handlers
        self._logger.addHandler(api_handler)
        self._logger.addHandler(prediction_handler)
        
        print(f"Logging initialized in: {self.log_dir}")
    
    def get_session(self) -> Optional[Session]:
        """Get database session."""
        if not SQLALCHEMY_AVAILABLE or not self.SessionLocal:
            return None
        
        try:
            return self.SessionLocal()
        except Exception as e:
            print(f"Failed to create database session: {e}")
            return None
    
    def create_user_session(self, ip_address: str = None, user_agent: str = None) -> Optional[str]:
        """
        Create a new user session.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            Session ID string or None if failed
        """
        if not SQLALCHEMY_AVAILABLE:
            return None
        
        session = self.get_session()
        if not session:
            return None
        
        try:
            user_session = UserSession(
                ip_address=ip_address,
                user_agent=user_agent
            )
            session.add(user_session)
            session.commit()
            session_id = user_session.session_id
            session.close()
            return session_id
        except Exception as e:
            session.rollback()
            session.close()
            print(f"Failed to create user session: {e}")
            return None
    
    def save_prediction(self,
                       prediction_type: str,
                       model_type: str,
                       predicted_label: str,
                       predicted_probability: float,
                       all_probabilities: Dict[str, float],
                       input_features: List[float] = None,
                       feature_names: List[str] = None,
                       time_series: List[float] = None,
                       flux_series: List[float] = None,
                       session_id: str = None,
                       dataset_id: int = None,
                       model_version: str = None,
                       confidence_level: str = None,
                       prediction_time_ms: float = None,
                       preprocessing_time_ms: float = None) -> Optional[int]:
        """
        Save prediction result to database.
        
        Args:
            prediction_type: Type of prediction ('features', 'lightcurve', 'upload')
            model_type: Model used for prediction
            predicted_label: Predicted class label
            predicted_probability: Probability of predicted class
            all_probabilities: All class probabilities
            input_features: Input feature array
            feature_names: Names of input features
            time_series: Time series for light curve data
            flux_series: Flux series for light curve data
            session_id: User session ID
            dataset_id: Associated dataset ID
            model_version: Version of the model used
            confidence_level: Confidence level of prediction
            prediction_time_ms: Time taken for prediction
            preprocessing_time_ms: Time taken for preprocessing
            
        Returns:
            Prediction ID or None if failed
        """
        # Log prediction to file logger
        log_data = {
            'prediction_type': prediction_type,
            'model_type': model_type,
            'predicted_label': predicted_label,
            'predicted_probability': predicted_probability,
            'all_probabilities': all_probabilities,
            'confidence_level': confidence_level,
            'prediction_time_ms': prediction_time_ms,
            'session_id': session_id
        }
        
        if self._logger:
            self._logger.info(f"PREDICTION: {json.dumps(log_data)}")
        
        # Save to database if available
        if SQLALCHEMY_AVAILABLE and self.SessionLocal:
            return self._save_prediction_sqlalchemy(
                prediction_type, model_type, predicted_label, predicted_probability,
                all_probabilities, input_features, feature_names, time_series,
                flux_series, session_id, dataset_id, model_version, confidence_level,
                prediction_time_ms, preprocessing_time_ms
            )
        else:
            return self._save_prediction_fallback(
                prediction_type, model_type, predicted_label, predicted_probability,
                all_probabilities, input_features, session_id
            )
    
    def _save_prediction_sqlalchemy(self, *args) -> Optional[int]:
        """Save prediction using SQLAlchemy."""
        session = self.get_session()
        if not session:
            return None
        
        try:
            (prediction_type, model_type, predicted_label, predicted_probability,
             all_probabilities, input_features, feature_names, time_series,
             flux_series, session_id, dataset_id, model_version, confidence_level,
             prediction_time_ms, preprocessing_time_ms) = args
            
            # Determine confidence level if not provided
            if not confidence_level:
                confidence_level = self._calculate_confidence_level(predicted_probability)
            
            prediction = Prediction(
                prediction_type=prediction_type,
                model_type=model_type,
                predicted_label=predicted_label,
                predicted_probability=predicted_probability,
                all_probabilities=all_probabilities,
                input_features=input_features,
                feature_names=feature_names,
                time_series=time_series,
                flux_series=flux_series,
                session_id=self._get_session_db_id(session_id) if session_id else None,
                dataset_id=dataset_id,
                model_version=model_version,
                confidence_level=confidence_level,
                prediction_time_ms=prediction_time_ms,
                preprocessing_time_ms=preprocessing_time_ms
            )
            
            session.add(prediction)
            session.commit()
            prediction_id = prediction.id
            session.close()
            return prediction_id
            
        except Exception as e:
            session.rollback()
            session.close()
            print(f"Failed to save prediction: {e}")
            return None
    
    def _save_prediction_fallback(self, prediction_type, model_type, predicted_label,
                                predicted_probability, all_probabilities, input_features,
                                session_id) -> Optional[int]:
        """Save prediction using fallback SQLite."""
        try:
            conn = sqlite3.connect(self.fallback_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO prediction_logs (
                    timestamp, prediction_type, model_type, predicted_label,
                    predicted_probability, all_probabilities, input_data, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                prediction_type,
                model_type,
                predicted_label,
                predicted_probability,
                json.dumps(all_probabilities),
                json.dumps(input_features) if input_features else None,
                session_id
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return prediction_id
            
        except Exception as e:
            print(f"Failed to save prediction to fallback database: {e}")
            return None
    
    def log_api_request(self,
                       endpoint: str,
                       method: str,
                       status_code: int,
                       response_time_ms: float,
                       ip_address: str = None,
                       user_agent: str = None,
                       session_id: str = None,
                       request_size: int = None,
                       response_size: int = None,
                       error_message: str = None,
                       error_type: str = None) -> Optional[int]:
        """
        Log API request to database and file.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            ip_address: Client IP address
            user_agent: Client user agent
            session_id: User session ID
            request_size: Request body size in bytes
            response_size: Response body size in bytes
            error_message: Error message if any
            error_type: Type of error if any
            
        Returns:
            Request log ID or None if failed
        """
        # Log to file
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time_ms': response_time_ms,
            'ip_address': ip_address,
            'session_id': session_id,
            'error_message': error_message
        }
        
        if self._logger:
            log_level = logging.ERROR if status_code >= 400 else logging.INFO
            self._logger.log(log_level, f"API_REQUEST: {json.dumps(log_data)}")
        
        # Save to database if available
        if SQLALCHEMY_AVAILABLE and self.SessionLocal:
            return self._log_api_request_sqlalchemy(
                endpoint, method, status_code, response_time_ms, ip_address,
                user_agent, session_id, request_size, response_size,
                error_message, error_type
            )
        else:
            return self._log_api_request_fallback(
                endpoint, method, status_code, response_time_ms, ip_address, error_message
            )
    
    def _log_api_request_sqlalchemy(self, *args) -> Optional[int]:
        """Log API request using SQLAlchemy."""
        session = self.get_session()
        if not session:
            return None
        
        try:
            (endpoint, method, status_code, response_time_ms, ip_address,
             user_agent, session_id, request_size, response_size,
             error_message, error_type) = args
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
            
            api_request = APIRequest(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=self._get_session_db_id(session_id) if session_id else None,
                request_size=request_size,
                response_size=response_size,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage,
                error_message=error_message,
                error_type=error_type
            )
            
            session.add(api_request)
            session.commit()
            request_id = api_request.id
            session.close()
            return request_id
            
        except Exception as e:
            session.rollback()
            session.close()
            print(f"Failed to log API request: {e}")
            return None
    
    def _log_api_request_fallback(self, endpoint, method, status_code,
                                response_time_ms, ip_address, error_message) -> Optional[int]:
        """Log API request using fallback SQLite."""
        try:
            conn = sqlite3.connect(self.fallback_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO api_request_logs (
                    timestamp, endpoint, method, status_code, response_time_ms,
                    ip_address, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                endpoint,
                method,
                status_code,
                response_time_ms,
                ip_address,
                error_message
            ))
            
            request_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return request_id
            
        except Exception as e:
            print(f"Failed to log API request to fallback database: {e}")
            return None
    
    def _get_session_db_id(self, session_id: str) -> Optional[int]:
        """Get database ID for session string ID."""
        if not SQLALCHEMY_AVAILABLE or not session_id:
            return None
        
        session = self.get_session()
        if not session:
            return None
        
        try:
            user_session = session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            
            if user_session:
                db_id = user_session.id
                session.close()
                return db_id
            else:
                session.close()
                return None
                
        except Exception as e:
            session.close()
            print(f"Failed to get session DB ID: {e}")
            return None
    
    def _calculate_confidence_level(self, probability: float) -> str:
        """Calculate confidence level based on probability."""
        if probability >= 0.8:
            return "HIGH"
        elif probability >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_prediction_history(self, session_id: str = None, limit: int = 100) -> List[Dict]:
        """
        Get prediction history for a session or all predictions.
        
        Args:
            session_id: User session ID (optional)
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        if not SQLALCHEMY_AVAILABLE:
            return []
        
        session = self.get_session()
        if not session:
            return []
        
        try:
            query = session.query(Prediction)
            
            if session_id:
                db_session_id = self._get_session_db_id(session_id)
                if db_session_id:
                    query = query.filter(Prediction.session_id == db_session_id)
            
            predictions = query.order_by(Prediction.created_at.desc()).limit(limit).all()
            result = [pred.to_dict() for pred in predictions]
            session.close()
            return result
            
        except Exception as e:
            session.close()
            print(f"Failed to get prediction history: {e}")
            return []
    
    def close(self):
        """Close database connections and cleanup."""
        if self.engine:
            self.engine.dispose()
        
        if self._logger:
            for handler in self._logger.handlers:
                handler.close()


# Global database manager instance
db_manager = None


def init_db(database_url: str = None, log_dir: str = None) -> DatabaseManager:
    """
    Initialize the global database manager.
    
    Args:
        database_url: Database connection string
        log_dir: Directory for log files
        
    Returns:
        DatabaseManager instance
    """
    global db_manager
    db_manager = DatabaseManager(database_url, log_dir)
    return db_manager


def get_db() -> DatabaseManager:
    """Get the global database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def save_prediction(prediction_type: str,
                   model_type: str,
                   predicted_label: str,
                   predicted_probability: float,
                   all_probabilities: Dict[str, float],
                   **kwargs) -> Optional[int]:
    """
    Convenience function to save prediction using global database manager.
    
    Args:
        prediction_type: Type of prediction
        model_type: Model used
        predicted_label: Predicted class
        predicted_probability: Prediction probability
        all_probabilities: All class probabilities
        **kwargs: Additional arguments for save_prediction
        
    Returns:
        Prediction ID or None
    """
    db = get_db()
    return db.save_prediction(
        prediction_type, model_type, predicted_label,
        predicted_probability, all_probabilities, **kwargs
    )


def log_api_request(endpoint: str,
                   method: str,
                   status_code: int,
                   response_time_ms: float,
                   **kwargs) -> Optional[int]:
    """
    Convenience function to log API request using global database manager.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        **kwargs: Additional arguments for log_api_request
        
    Returns:
        Request log ID or None
    """
    db = get_db()
    return db.log_api_request(endpoint, method, status_code, response_time_ms, **kwargs)
