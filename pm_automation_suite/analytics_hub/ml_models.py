"""
Machine Learning Models for PM Performance Analytics

Includes predictive models for PM performance, burnout risk detection,
productivity analysis, and project success prediction using advanced ML techniques.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


class PerformanceMetric(Enum):
    """PM performance metrics to predict."""
    VELOCITY = "velocity"
    QUALITY = "quality"
    BURNOUT_RISK = "burnout_risk"
    PRODUCTIVITY = "productivity"
    PROJECT_SUCCESS = "project_success"
    DEADLINE_ADHERENCE = "deadline_adherence"
    TEAM_SATISFACTION = "team_satisfaction"


@dataclass
class ModelMetrics:
    """Metrics for model performance."""
    model_name: str
    train_score: float
    validation_score: float
    test_score: Optional[float] = None
    cross_val_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict[str, Any]] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'train_score': self.train_score,
            'validation_score': self.validation_score,
            'test_score': self.test_score,
            'cross_val_scores': self.cross_val_scores,
            'feature_importance': self.feature_importance,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'classification_report': self.classification_report,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }


@dataclass
class PredictionResult:
    """Result of a model prediction."""
    prediction: Union[float, str, np.ndarray]
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    feature_contributions: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'prediction': self.prediction.tolist() if isinstance(self.prediction, np.ndarray) else self.prediction,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'feature_contributions': self.feature_contributions,
            'metadata': self.metadata
        }


class BaseMLModel:
    """Base class for ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model_type = ModelType(config.get('model_type', 'regression'))
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.model_path = Path(config.get('model_path', './models'))
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    def preprocess_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None, fit: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for model training/prediction.
        
        Args:
            X: Feature data
            y: Target data (optional)
            fit: Whether to fit the preprocessors
            
        Returns:
            Preprocessed features and targets
        """
        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
        
        # Scale features
        if self.scaler is None and fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Process target if provided
        y_processed = None
        if y is not None:
            y_processed = y.values
        
        return X_scaled, y_processed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Train the model."""
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X: pd.DataFrame) -> PredictionResult:
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate model performance."""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def save_model(self, model_name: str):
        """Save model to disk."""
        model_file = self.model_path / f"{model_name}.pkl"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {model_file}")
    
    def load_model(self, model_name: str):
        """Load model from disk."""
        model_file = self.model_path / f"{model_name}.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from: {model_file}")


class PMPerformancePredictor(BaseMLModel):
    """
    Predicts PM performance metrics including velocity, quality, and deadline adherence.
    
    Uses ensemble methods and feature engineering specific to PM activities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PM Performance Predictor."""
        super().__init__(config)
        self.metric_to_predict = PerformanceMetric(config.get('metric', 'velocity'))
        self.use_ensemble = config.get('use_ensemble', True)
        
    def engineer_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features specific to PM performance.
        
        Args:
            raw_data: Raw PM activity data
            
        Returns:
            Feature engineered dataframe
        """
        features = pd.DataFrame()
        
        # Time-based features
        if 'timestamp' in raw_data.columns:
            raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
            features['hour_of_day'] = raw_data['timestamp'].dt.hour
            features['day_of_week'] = raw_data['timestamp'].dt.dayofweek
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
            features['month'] = raw_data['timestamp'].dt.month
            features['quarter'] = raw_data['timestamp'].dt.quarter
        
        # Workload features
        if 'tickets_assigned' in raw_data.columns:
            features['tickets_assigned'] = raw_data['tickets_assigned']
            features['tickets_assigned_rolling_7d'] = raw_data['tickets_assigned'].rolling(7, min_periods=1).mean()
            features['tickets_assigned_rolling_30d'] = raw_data['tickets_assigned'].rolling(30, min_periods=1).mean()
        
        if 'tickets_completed' in raw_data.columns:
            features['tickets_completed'] = raw_data['tickets_completed']
            features['completion_rate'] = raw_data['tickets_completed'] / (raw_data['tickets_assigned'] + 1)
            features['completion_rate_rolling_7d'] = features['completion_rate'].rolling(7, min_periods=1).mean()
        
        # Meeting and communication features
        if 'meetings_count' in raw_data.columns:
            features['meetings_count'] = raw_data['meetings_count']
            features['meetings_hours'] = raw_data.get('meetings_hours', 0)
            features['meeting_efficiency'] = features['meetings_count'] / (features['meetings_hours'] + 1)
        
        if 'emails_sent' in raw_data.columns:
            features['emails_sent'] = raw_data['emails_sent']
            features['emails_received'] = raw_data.get('emails_received', 0)
            features['email_ratio'] = features['emails_sent'] / (features['emails_received'] + 1)
        
        # Team features
        if 'team_size' in raw_data.columns:
            features['team_size'] = raw_data['team_size']
            features['team_velocity'] = raw_data.get('team_velocity', 0)
            features['individual_contribution'] = raw_data.get('tickets_completed', 0) / (features['team_velocity'] + 1)
        
        # Complexity features
        if 'story_points' in raw_data.columns:
            features['avg_story_points'] = raw_data['story_points']
            features['story_point_variance'] = raw_data['story_points'].rolling(7, min_periods=1).std()
        
        # Quality features
        if 'bugs_reported' in raw_data.columns:
            features['bugs_reported'] = raw_data['bugs_reported']
            features['bug_rate'] = features['bugs_reported'] / (raw_data.get('tickets_completed', 1) + 1)
        
        # Deadline features
        if 'on_time_delivery' in raw_data.columns:
            features['on_time_delivery_rate'] = raw_data['on_time_delivery']
            features['on_time_rolling_30d'] = features['on_time_delivery_rate'].rolling(30, min_periods=1).mean()
        
        # Interaction features
        features['workload_complexity'] = features.get('tickets_assigned', 0) * features.get('avg_story_points', 1)
        features['communication_load'] = features.get('meetings_hours', 0) + features.get('emails_sent', 0) / 10
        
        # Lag features
        for lag in [1, 7, 30]:
            if 'tickets_completed' in features.columns:
                features[f'tickets_completed_lag_{lag}'] = features['tickets_completed'].shift(lag)
        
        # Fill missing values
        features = features.fillna(0)
        
        return features
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Train the PM performance prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for model training")
        
        logger.info(f"Training PM Performance Predictor for metric: {self.metric_to_predict.value}")
        
        start_time = datetime.now()
        
        # Engineer features
        X_train_eng = self.engineer_features(X_train)
        X_val_eng = self.engineer_features(X_val) if X_val is not None else None
        
        # Preprocess data
        X_train_scaled, y_train_processed = self.preprocess_data(X_train_eng, y_train, fit=True)
        X_val_scaled = None
        if X_val_eng is not None:
            X_val_scaled, _ = self.preprocess_data(X_val_eng)
        
        # Train model based on metric type
        if self.metric_to_predict in [PerformanceMetric.VELOCITY, PerformanceMetric.PRODUCTIVITY]:
            # Regression task
            if self.use_ensemble:
                # Train multiple models and ensemble
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                
                rf_model.fit(X_train_scaled, y_train_processed)
                gb_model.fit(X_train_scaled, y_train_processed)
                
                # Simple averaging ensemble
                self.model = {
                    'rf': rf_model,
                    'gb': gb_model,
                    'weights': [0.5, 0.5]
                }
            else:
                self.model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                self.model.fit(X_train_scaled, y_train_processed)
        
        elif self.metric_to_predict in [PerformanceMetric.BURNOUT_RISK, PerformanceMetric.PROJECT_SUCCESS]:
            # Classification task
            if self.use_ensemble and XGBOOST_AVAILABLE:
                # Use XGBoost for better performance
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                self.model.fit(X_train_scaled, y_train_processed)
            else:
                self.model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                self.model.fit(X_train_scaled, y_train_processed)
        
        self.is_trained = True
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Model training completed in {training_time:.2f}s")
        
        # Validate if validation data provided
        if X_val_scaled is not None and y_val is not None:
            val_predictions = self._predict_raw(X_val_scaled)
            if self.model_type == ModelType.REGRESSION:
                val_score = 1 - mean_squared_error(y_val, val_predictions) / np.var(y_val)
            else:
                val_score = accuracy_score(y_val, val_predictions)
            
            logger.info(f"Validation score: {val_score:.4f}")
    
    def _predict_raw(self, X_scaled: np.ndarray) -> np.ndarray:
        """Make raw predictions with scaled features."""
        if isinstance(self.model, dict) and 'rf' in self.model:
            # Ensemble prediction
            rf_pred = self.model['rf'].predict(X_scaled)
            gb_pred = self.model['gb'].predict(X_scaled)
            weights = self.model['weights']
            return weights[0] * rf_pred + weights[1] * gb_pred
        else:
            return self.model.predict(X_scaled)
    
    def predict(self, X: pd.DataFrame) -> PredictionResult:
        """
        Make PM performance predictions.
        
        Args:
            X: Feature data
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Engineer features
        X_eng = self.engineer_features(X)
        
        # Preprocess
        X_scaled, _ = self.preprocess_data(X_eng)
        
        # Make predictions
        predictions = self._predict_raw(X_scaled)
        
        # Calculate feature importance
        feature_contributions = {}
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for i, feature in enumerate(self.feature_names):
                if i < len(importances):
                    feature_contributions[feature] = float(importances[i])
        
        # Create result
        result = PredictionResult(
            prediction=float(predictions[0]) if len(predictions) == 1 else predictions,
            feature_contributions=feature_contributions,
            metadata={
                'metric': self.metric_to_predict.value,
                'model_type': 'ensemble' if self.use_ensemble else 'single',
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return result
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Model performance metrics
        """
        # Engineer features
        X_test_eng = self.engineer_features(X_test)
        
        # Preprocess
        X_test_scaled, _ = self.preprocess_data(X_test_eng)
        
        # Make predictions
        predictions = self._predict_raw(X_test_scaled)
        
        # Calculate metrics
        if self.model_type == ModelType.REGRESSION:
            mse = mean_squared_error(y_test, predictions)
            test_score = 1 - mse / np.var(y_test)  # R-squared
        else:
            test_score = accuracy_score(y_test, predictions)
        
        # Get feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for i, feature in enumerate(self.feature_names):
                if i < len(importances):
                    feature_importance[feature] = float(importances[i])
        
        metrics = ModelMetrics(
            model_name="PMPerformancePredictor",
            train_score=0.0,  # Would need to recalculate on training data
            validation_score=0.0,  # Would need validation data
            test_score=test_score,
            feature_importance=feature_importance
        )
        
        return metrics


class BurnoutPredictor(BaseMLModel):
    """
    Predicts PM burnout risk using behavioral patterns and workload indicators.
    
    Uses deep learning for complex pattern recognition in time-series data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Burnout Predictor."""
        super().__init__(config)
        self.sequence_length = config.get('sequence_length', 30)  # Days of history
        self.use_lstm = config.get('use_lstm', True) and TENSORFLOW_AVAILABLE
        self.burnout_threshold = config.get('burnout_threshold', 0.7)
        
    def create_sequences(self, data: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series prediction.
        
        Args:
            data: Feature data
            target: Target labels
            
        Returns:
            Sequences and corresponding targets
        """
        sequences = []
        targets = []
        
        data_array = data.values
        target_array = target.values
        
        for i in range(len(data) - self.sequence_length):
            seq = data_array[i:i + self.sequence_length]
            label = target_array[i + self.sequence_length]
            sequences.append(seq)
            targets.append(label)
        
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]):
        """
        Build LSTM model for burnout prediction.
        
        Args:
            input_shape: Shape of input sequences
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(16),
            layers.Dropout(0.2),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def extract_burnout_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features specific to burnout detection.
        
        Args:
            raw_data: Raw activity data
            
        Returns:
            Burnout-specific features
        """
        features = pd.DataFrame()
        
        # Work hours patterns
        if 'work_hours' in raw_data.columns:
            features['daily_work_hours'] = raw_data['work_hours']
            features['work_hours_7d_avg'] = features['daily_work_hours'].rolling(7, min_periods=1).mean()
            features['work_hours_variance'] = features['daily_work_hours'].rolling(7, min_periods=1).std()
            features['excessive_hours'] = (features['daily_work_hours'] > 10).astype(int)
            features['weekend_work'] = raw_data.get('weekend_work_hours', 0)
        
        # Meeting load
        if 'meetings_count' in raw_data.columns:
            features['meeting_overload'] = (raw_data['meetings_count'] > 6).astype(int)
            features['meeting_hours_ratio'] = raw_data.get('meetings_hours', 0) / (raw_data.get('work_hours', 8) + 1)
        
        # Task switching and interruptions
        if 'task_switches' in raw_data.columns:
            features['task_switches'] = raw_data['task_switches']
            features['interruption_rate'] = raw_data.get('interruptions', 0) / (raw_data.get('work_hours', 8) + 1)
        
        # Deadline pressure
        if 'overdue_tasks' in raw_data.columns:
            features['overdue_tasks'] = raw_data['overdue_tasks']
            features['deadline_pressure'] = raw_data.get('urgent_tasks', 0) / (raw_data.get('total_tasks', 1) + 1)
        
        # Communication patterns
        if 'response_time' in raw_data.columns:
            features['avg_response_time'] = raw_data['response_time']
            features['delayed_responses'] = (features['avg_response_time'] > 240).astype(int)  # >4 hours
        
        # Break patterns
        if 'break_time' in raw_data.columns:
            features['break_time_minutes'] = raw_data['break_time']
            features['insufficient_breaks'] = (features['break_time_minutes'] < 30).astype(int)
        
        # Stress indicators
        features['stress_score'] = (
            features.get('excessive_hours', 0) * 0.3 +
            features.get('meeting_overload', 0) * 0.2 +
            features.get('delayed_responses', 0) * 0.2 +
            features.get('insufficient_breaks', 0) * 0.3
        )
        
        # Recovery indicators
        if 'vacation_days' in raw_data.columns:
            features['days_since_vacation'] = raw_data['days_since_vacation']
            features['vacation_deficit'] = (features['days_since_vacation'] > 90).astype(int)
        
        return features
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Train burnout prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets (1 = burnout risk, 0 = healthy)
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Training Burnout Predictor")
        
        # Extract burnout features
        X_train_features = self.extract_burnout_features(X_train)
        X_val_features = self.extract_burnout_features(X_val) if X_val is not None else None
        
        if self.use_lstm and TENSORFLOW_AVAILABLE:
            # Prepare sequences for LSTM
            X_train_seq, y_train_seq = self.create_sequences(X_train_features, y_train)
            
            # Build and train LSTM model
            input_shape = (self.sequence_length, X_train_features.shape[1])
            self.model = self.build_lstm_model(input_shape)
            
            # Train with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.feature_names = X_train_features.columns.tolist()
            
        else:
            # Fall back to traditional ML
            X_train_scaled, y_train_processed = self.preprocess_data(X_train_features, y_train, fit=True)
            
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    scale_pos_weight=3,  # Handle class imbalance
                    random_state=42
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            
            self.model.fit(X_train_scaled, y_train_processed)
        
        self.is_trained = True
        logger.info("Burnout Predictor training completed")
    
    def predict(self, X: pd.DataFrame) -> PredictionResult:
        """
        Predict burnout risk.
        
        Args:
            X: Feature data
            
        Returns:
            Burnout risk prediction
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        X_features = self.extract_burnout_features(X)
        
        if self.use_lstm and isinstance(self.model, keras.Model):
            # Prepare sequence
            if len(X_features) < self.sequence_length:
                # Pad sequence if too short
                padding = pd.DataFrame(
                    np.zeros((self.sequence_length - len(X_features), X_features.shape[1])),
                    columns=X_features.columns
                )
                X_features = pd.concat([padding, X_features], ignore_index=True)
            
            X_seq = X_features.values[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Predict
            burnout_probability = float(self.model.predict(X_seq, verbose=0)[0][0])
            
        else:
            # Traditional ML prediction
            X_scaled, _ = self.preprocess_data(X_features.tail(1))
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
                burnout_probability = probabilities[1]  # Probability of burnout
            else:
                prediction = self.model.predict(X_scaled)[0]
                burnout_probability = float(prediction)
        
        # Determine risk level
        if burnout_probability >= self.burnout_threshold:
            risk_level = "HIGH"
        elif burnout_probability >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Get key risk factors
        risk_factors = self._identify_risk_factors(X_features.iloc[-1])
        
        result = PredictionResult(
            prediction=risk_level,
            confidence=burnout_probability,
            probabilities={
                'healthy': 1 - burnout_probability,
                'burnout_risk': burnout_probability
            },
            metadata={
                'risk_factors': risk_factors,
                'threshold': self.burnout_threshold,
                'model_type': 'lstm' if self.use_lstm else 'traditional'
            }
        )
        
        return result
    
    def _identify_risk_factors(self, features: pd.Series) -> List[str]:
        """Identify key burnout risk factors from features."""
        risk_factors = []
        
        if features.get('excessive_hours', 0) > 0:
            risk_factors.append("Excessive work hours (>10 hours/day)")
        
        if features.get('work_hours_variance', 0) > 2:
            risk_factors.append("Irregular work schedule")
        
        if features.get('weekend_work', 0) > 4:
            risk_factors.append("Frequent weekend work")
        
        if features.get('meeting_overload', 0) > 0:
            risk_factors.append("Too many meetings (>6/day)")
        
        if features.get('insufficient_breaks', 0) > 0:
            risk_factors.append("Insufficient break time")
        
        if features.get('vacation_deficit', 0) > 0:
            risk_factors.append("No vacation in 90+ days")
        
        if features.get('deadline_pressure', 0) > 0.5:
            risk_factors.append("High deadline pressure")
        
        return risk_factors


class ProductivityAnalyzer(BaseMLModel):
    """
    Analyzes and predicts PM productivity patterns using multivariate analysis.
    
    Identifies productivity drivers and provides actionable recommendations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Productivity Analyzer."""
        super().__init__(config)
        self.productivity_components = config.get('components', [
            'task_completion', 'quality_score', 'collaboration_index', 'innovation_score'
        ])
        self.use_clustering = config.get('use_clustering', True)
        
    def calculate_productivity_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate composite productivity score.
        
        Args:
            data: PM activity data
            
        Returns:
            Productivity scores
        """
        score = pd.Series(index=data.index, dtype=float)
        
        # Task completion component (40%)
        if 'tasks_completed' in data.columns and 'tasks_assigned' in data.columns:
            completion_rate = data['tasks_completed'] / (data['tasks_assigned'] + 1)
            score += 0.4 * completion_rate
        
        # Quality component (30%)
        if 'bugs_reported' in data.columns and 'tasks_completed' in data.columns:
            quality_score = 1 - (data['bugs_reported'] / (data['tasks_completed'] + 1))
            score += 0.3 * quality_score.clip(0, 1)
        
        # Efficiency component (20%)
        if 'story_points_completed' in data.columns and 'work_hours' in data.columns:
            efficiency = data['story_points_completed'] / (data['work_hours'] + 1)
            normalized_efficiency = efficiency / efficiency.max() if efficiency.max() > 0 else efficiency
            score += 0.2 * normalized_efficiency
        
        # Collaboration component (10%)
        if 'pr_reviews' in data.columns or 'knowledge_shares' in data.columns:
            collaboration = data.get('pr_reviews', 0) + data.get('knowledge_shares', 0)
            normalized_collaboration = collaboration / collaboration.max() if collaboration.max() > 0 else collaboration
            score += 0.1 * normalized_collaboration
        
        return score * 100  # Convert to percentage
    
    def identify_productivity_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify patterns in productivity data.
        
        Args:
            data: Historical productivity data
            
        Returns:
            Identified patterns and insights
        """
        patterns = {
            'daily_patterns': {},
            'weekly_patterns': {},
            'seasonal_patterns': {},
            'correlation_insights': {}
        }
        
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['month'] = data['timestamp'].dt.month
            
            # Daily productivity patterns
            hourly_productivity = data.groupby('hour')['productivity_score'].mean()
            patterns['daily_patterns'] = {
                'peak_hours': hourly_productivity.nlargest(3).index.tolist(),
                'low_hours': hourly_productivity.nsmallest(3).index.tolist(),
                'average_by_hour': hourly_productivity.to_dict()
            }
            
            # Weekly patterns
            daily_productivity = data.groupby('day_of_week')['productivity_score'].mean()
            patterns['weekly_patterns'] = {
                'best_days': daily_productivity.nlargest(2).index.tolist(),
                'worst_days': daily_productivity.nsmallest(2).index.tolist(),
                'average_by_day': daily_productivity.to_dict()
            }
        
        # Correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlations = data[numeric_cols].corr()['productivity_score'].sort_values(ascending=False)
        
        patterns['correlation_insights'] = {
            'positive_factors': correlations[correlations > 0.3].index.tolist(),
            'negative_factors': correlations[correlations < -0.3].index.tolist(),
            'correlations': correlations.to_dict()
        }
        
        return patterns
    
    def generate_recommendations(self, current_metrics: Dict[str, float], patterns: Dict[str, Any]) -> List[str]:
        """
        Generate productivity improvement recommendations.
        
        Args:
            current_metrics: Current productivity metrics
            patterns: Identified productivity patterns
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Time management recommendations
        if 'daily_patterns' in patterns:
            peak_hours = patterns['daily_patterns'].get('peak_hours', [])
            if peak_hours:
                recommendations.append(
                    f"Schedule important tasks during peak productivity hours: {', '.join(map(str, peak_hours))}"
                )
        
        # Workload recommendations
        current_completion_rate = current_metrics.get('completion_rate', 0)
        if current_completion_rate < 0.7:
            recommendations.append(
                "Consider reducing concurrent task assignments to improve completion rate"
            )
        
        # Quality recommendations
        current_bug_rate = current_metrics.get('bug_rate', 0)
        if current_bug_rate > 0.1:
            recommendations.append(
                "Allocate more time for code review and testing to reduce bug rate"
            )
        
        # Meeting optimization
        meeting_hours_ratio = current_metrics.get('meeting_hours_ratio', 0)
        if meeting_hours_ratio > 0.4:
            recommendations.append(
                "Meeting time exceeds 40% of work hours - consider consolidating or eliminating meetings"
            )
        
        # Break recommendations
        if current_metrics.get('break_time_minutes', 0) < 30:
            recommendations.append(
                "Take regular breaks (at least 5 minutes every hour) to maintain productivity"
            )
        
        # Collaboration recommendations
        if 'positive_factors' in patterns.get('correlation_insights', {}):
            positive_factors = patterns['correlation_insights']['positive_factors']
            if 'pr_reviews' in positive_factors:
                recommendations.append(
                    "Increase code review participation - strong positive correlation with productivity"
                )
        
        return recommendations
    
    def predict(self, X: pd.DataFrame) -> PredictionResult:
        """
        Predict future productivity and provide recommendations.
        
        Args:
            X: Current and historical data
            
        Returns:
            Productivity prediction and recommendations
        """
        # Calculate current productivity score
        current_productivity = self.calculate_productivity_score(X.tail(1)).iloc[0]
        
        # Identify patterns
        patterns = self.identify_productivity_patterns(X)
        
        # Extract current metrics
        current_metrics = X.tail(1).to_dict('records')[0]
        
        # Generate recommendations
        recommendations = self.generate_recommendations(current_metrics, patterns)
        
        # Predict productivity trend
        if len(X) > 7:
            recent_scores = self.calculate_productivity_score(X.tail(7))
            trend = "improving" if recent_scores.iloc[-1] > recent_scores.mean() else "declining"
        else:
            trend = "insufficient_data"
        
        result = PredictionResult(
            prediction=current_productivity,
            metadata={
                'trend': trend,
                'patterns': patterns,
                'recommendations': recommendations,
                'components': {
                    'task_completion': current_metrics.get('completion_rate', 0) * 100,
                    'quality': (1 - current_metrics.get('bug_rate', 0)) * 100,
                    'efficiency': current_metrics.get('efficiency_score', 0) * 100
                }
            }
        )
        
        return result