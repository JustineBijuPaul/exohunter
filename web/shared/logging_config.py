"""
Advanced logging configuration for ExoHunter web applications.
Provides structured logging with file rotation, request tracking, and performance metrics.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'user_agent'):
            log_data['user_agent'] = record.user_agent
        if hasattr(record, 'processing_time'):
            log_data['processing_time_ms'] = record.processing_time
        
        return json.dumps(log_data)


class RequestLogger:
    """Logger for API request tracking."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize request logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('request_logger')
        self.logger.setLevel(logging.INFO)
        
        # Request log with rotation
        request_handler = RotatingFileHandler(
            self.log_dir / 'requests.log',
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        request_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(request_handler)
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        processing_time_ms: float,
        request_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        **kwargs
    ):
        """
        Log API request.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            processing_time_ms: Processing time in milliseconds
            request_id: Optional request ID
            user_agent: Optional user agent string
            **kwargs: Additional fields to log
        """
        extra = {
            'request_id': request_id,
            'user_agent': user_agent,
            'processing_time': processing_time_ms
        }
        
        self.logger.info(
            f"{method} {path} - {status_code} - {processing_time_ms:.2f}ms",
            extra=extra
        )


class PredictionLogger:
    """Logger for prediction tracking."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize prediction logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('prediction_logger')
        self.logger.setLevel(logging.INFO)
        
        # Prediction log with daily rotation
        prediction_handler = TimedRotatingFileHandler(
            self.log_dir / 'predictions.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        prediction_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(prediction_handler)
    
    def log_prediction(
        self,
        predictions: Dict[str, str],
        confidences: Dict[str, float],
        ensemble_prediction: str,
        ensemble_confidence: float,
        processing_time_ms: float,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """
        Log prediction results.
        
        Args:
            predictions: Model predictions
            confidences: Confidence scores
            ensemble_prediction: Ensemble prediction result
            ensemble_confidence: Ensemble confidence score
            processing_time_ms: Processing time in milliseconds
            request_id: Optional request ID
            **kwargs: Additional fields to log
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'predictions': predictions,
            'confidences': {k: round(v, 2) for k, v in confidences.items()},
            'ensemble_prediction': ensemble_prediction,
            'ensemble_confidence': round(ensemble_confidence, 2),
            'processing_time_ms': round(processing_time_ms, 2),
            **kwargs
        }
        
        self.logger.info(
            f"Prediction: {ensemble_prediction} ({ensemble_confidence:.1f}% confidence)",
            extra={'request_id': request_id, 'processing_time': processing_time_ms}
        )


class PerformanceTracker:
    """Track and log performance metrics."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize performance tracker.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / 'performance_metrics.json'
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics or create new."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'total_requests': 0,
            'total_predictions': 0,
            'avg_response_time_ms': 0.0,
            'prediction_distribution': {},
            'error_count': 0,
            'start_time': datetime.now().isoformat()
        }
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save metrics: {e}")
    
    def track_request(self, processing_time_ms: float):
        """Track request metrics."""
        self.metrics['total_requests'] += 1
        
        # Update running average
        n = self.metrics['total_requests']
        old_avg = self.metrics['avg_response_time_ms']
        self.metrics['avg_response_time_ms'] = (
            (old_avg * (n - 1) + processing_time_ms) / n
        )
        
        self._save_metrics()
    
    def track_prediction(self, prediction: str):
        """Track prediction distribution."""
        self.metrics['total_predictions'] += 1
        
        if 'prediction_distribution' not in self.metrics:
            self.metrics['prediction_distribution'] = {}
        
        dist = self.metrics['prediction_distribution']
        dist[prediction] = dist.get(prediction, 0) + 1
        
        self._save_metrics()
    
    def track_error(self):
        """Track error count."""
        self.metrics['error_count'] += 1
        self._save_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            'total_requests': 0,
            'total_predictions': 0,
            'avg_response_time_ms': 0.0,
            'prediction_distribution': {},
            'error_count': 0,
            'start_time': datetime.now().isoformat()
        }
        self._save_metrics()


def setup_logging(
    app_name: str = "exohunter",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    json_format: bool = False
) -> logging.Logger:
    """
    Setup application logging.
    
    Args:
        app_name: Name of the application
        log_dir: Directory for log files (default: logs/)
        level: Logging level
        json_format: Use JSON formatting
    
    Returns:
        Configured logger
    """
    if log_dir is None:
        log_dir = Path('logs')
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / f'{app_name}.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    
    # Set formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Global loggers (initialized by applications)
request_logger: Optional[RequestLogger] = None
prediction_logger: Optional[PredictionLogger] = None
performance_tracker: Optional[PerformanceTracker] = None


def initialize_loggers(log_dir: Path = Path('logs')):
    """
    Initialize all global loggers.
    
    Args:
        log_dir: Directory for log files
    """
    global request_logger, prediction_logger, performance_tracker
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    request_logger = RequestLogger(log_dir)
    prediction_logger = PredictionLogger(log_dir)
    performance_tracker = PerformanceTracker(log_dir)
    
    logging.info(f"Initialized logging system in {log_dir.absolute()}")
