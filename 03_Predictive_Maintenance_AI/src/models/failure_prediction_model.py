import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from typing import Dict, List, Tuple, Any
import logging

class FailurePredictionModel:
    """
    Advanced machine learning model for predicting equipment failures
    in manufacturing environments
    """

    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.training_history = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, data: pd.DataFrame, target_column: str = 'failure_within_days') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and preprocess data for model training

        Args:
            data: Raw sensor data with features and target
            target_column: Name of the target column

        Returns:
            Tuple of (features, targets)
        """

        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = self.label_encoder.fit_transform(X[col].astype(str))

        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)

        # Encode target if necessary
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values

        self.logger.info(f"Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

        return X_scaled, y_encoded

    def create_ensemble_model(self) -> object:
        """Create ensemble model combining multiple algorithms"""

        from sklearn.ensemble import VotingClassifier

        # Individual models
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )

        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            voting='soft'
        )

        return ensemble

    def create_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create LSTM model for time series prediction"""

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def train_model(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the predictive maintenance model

        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Fraction of data for validation

        Returns:
            Training results dictionary
        """

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        if self.model_type == 'ensemble':
            model = self.create_ensemble_model()

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = model.feature_importances_

        elif self.model_type == 'lstm':
            # Reshape for LSTM (samples, timesteps, features)
            X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

            model = self.create_lstm_model((1, X_train.shape[1]))

            # Train model
            history = model.fit(
                X_train_lstm, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test_lstm, y_test),
                verbose=0
            )

            # Predictions
            y_pred_proba = model.predict(X_test_lstm).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            self.training_history = history.history

        self.model = model

        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'model_type': self.model_type
        }

        self.logger.info(f"Model trained successfully. Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")

        return results

    def predict_failure_probability(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """
        Predict failure probability for given sensor data

        Args:
            sensor_data: Sensor readings for prediction

        Returns:
            Prediction results with probability and risk level
        """

        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        # Scale input data
        sensor_data_scaled = self.scaler.transform(sensor_data.reshape(1, -1))

        if self.model_type == 'lstm':
            sensor_data_scaled = sensor_data_scaled.reshape((1, 1, sensor_data_scaled.shape[1]))

        # Make prediction
        failure_probability = self.model.predict_proba(sensor_data_scaled)[0, 1] if self.model_type == 'ensemble' else self.model.predict(sensor_data_scaled)[0, 0]

        # Determine risk level
        if failure_probability > 0.8:
            risk_level = 'CRITICAL'
            recommended_action = 'Immediate maintenance required'
        elif failure_probability > 0.6:
            risk_level = 'HIGH'
            recommended_action = 'Schedule maintenance within 24 hours'
        elif failure_probability > 0.4:
            risk_level = 'MEDIUM'
            recommended_action = 'Schedule maintenance within 1 week'
        elif failure_probability > 0.2:
            risk_level = 'LOW'
            recommended_action = 'Monitor closely, normal maintenance schedule'
        else:
            risk_level = 'NORMAL'
            recommended_action = 'Continue normal operation'

        return {
            'failure_probability': float(failure_probability),
            'risk_level': risk_level,
            'recommended_action': recommended_action,
            'confidence_score': self._calculate_confidence(sensor_data_scaled)
        }

    def _calculate_confidence(self, sensor_data: np.ndarray) -> float:
        """Calculate prediction confidence based on data quality"""

        # Simple confidence calculation based on data completeness and range
        # In production, would use more sophisticated methods

        confidence = 0.85  # Base confidence

        # Check for missing values or outliers
        if np.any(np.isnan(sensor_data)):
            confidence -= 0.2

        # Check if values are within expected ranges (simplified)
        if np.any(np.abs(sensor_data) > 3):  # Beyond 3 standard deviations
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk"""

        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk"""

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data['feature_importance']

        self.logger.info(f"Model loaded from {filepath}")

class MaintenanceScheduleOptimizer:
    """
    Optimize maintenance schedules based on failure predictions
    """

    def __init__(self):
        self.maintenance_costs = {
            'preventive': 1000,
            'corrective': 5000,
            'emergency': 15000
        }

        self.downtime_costs = {
            'planned': 500,  # per hour
            'unplanned': 2000  # per hour
        }

    def optimize_schedule(self, equipment_predictions: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Optimize maintenance schedule based on failure predictions

        Args:
            equipment_predictions: Dictionary of equipment IDs and their predictions

        Returns:
            Optimized maintenance schedule
        """

        schedule = []

        for equipment_id, prediction in equipment_predictions.items():
            failure_prob = prediction['failure_probability']
            risk_level = prediction['risk_level']

            # Determine optimal maintenance timing
            if risk_level == 'CRITICAL':
                days_until_maintenance = 1
                maintenance_type = 'emergency'
                priority = 1
            elif risk_level == 'HIGH':
                days_until_maintenance = 3
                maintenance_type = 'corrective'
                priority = 2
            elif risk_level == 'MEDIUM':
                days_until_maintenance = 7
                maintenance_type = 'preventive'
                priority = 3
            else:
                days_until_maintenance = 30
                maintenance_type = 'preventive'
                priority = 4

            # Calculate expected cost savings
            cost_without_maintenance = self._calculate_failure_cost(failure_prob)
            cost_with_maintenance = self.maintenance_costs[maintenance_type]
            expected_savings = cost_without_maintenance - cost_with_maintenance

            schedule.append({
                'equipment_id': equipment_id,
                'days_until_maintenance': days_until_maintenance,
                'maintenance_type': maintenance_type,
                'priority': priority,
                'failure_probability': failure_prob,
                'expected_cost_savings': max(0, expected_savings),
                'recommended_actions': self._get_maintenance_actions(risk_level)
            })

        # Sort by priority and expected savings
        schedule.sort(key=lambda x: (x['priority'], -x['expected_cost_savings']))

        return schedule

    def _calculate_failure_cost(self, failure_probability: float) -> float:
        """Calculate expected cost of failure"""

        # Expected cost = probability * cost of failure
        failure_cost = (
            self.maintenance_costs['emergency'] + 
            self.downtime_costs['unplanned'] * 8  # Assume 8 hours average downtime
        )

        return failure_probability * failure_cost

    def _get_maintenance_actions(self, risk_level: str) -> List[str]:
        """Get specific maintenance actions based on risk level"""

        actions = {
            'CRITICAL': [
                'Stop operation immediately',
                'Inspect critical components', 
                'Replace worn parts',
                'Perform full system check',
                'Update maintenance records'
            ],
            'HIGH': [
                'Schedule immediate inspection',
                'Prepare replacement parts',
                'Plan maintenance window',
                'Notify operations team',
                'Monitor continuously'
            ],
            'MEDIUM': [
                'Schedule routine maintenance',
                'Order replacement parts',
                'Plan maintenance resources',
                'Increase monitoring frequency'
            ],
            'LOW': [
                'Continue monitoring',
                'Plan preventive maintenance',
                'Stock standard parts'
            ],
            'NORMAL': [
                'Normal monitoring',
                'Routine maintenance schedule'
            ]
        }

        return actions.get(risk_level, ['Monitor equipment'])

# Utility functions for data generation and testing
def generate_sample_sensor_data(num_samples: int = 1000) -> pd.DataFrame:
    """Generate realistic sample sensor data for testing"""

    np.random.seed(42)

    # Simulate different sensor readings
    data = {
        'temperature': np.random.normal(65, 10, num_samples),  # °C
        'vibration_x': np.random.normal(2.5, 0.8, num_samples),  # mm/s
        'vibration_y': np.random.normal(2.2, 0.7, num_samples),
        'vibration_z': np.random.normal(1.8, 0.6, num_samples),
        'pressure': np.random.normal(150, 20, num_samples),  # PSI
        'flow_rate': np.random.normal(45, 8, num_samples),  # L/min
        'power_consumption': np.random.normal(2200, 300, num_samples),  # Watts
        'operating_hours': np.random.uniform(0, 8760, num_samples),  # Hours per year
        'maintenance_age': np.random.uniform(0, 180, num_samples),  # Days since last maintenance
    }

    df = pd.DataFrame(data)

    # Create realistic failure indicators
    # Higher chance of failure with:
    # - High temperature, vibration, power consumption
    # - Low pressure, flow rate
    # - High operating hours and maintenance age

    failure_score = (
        (df['temperature'] - 65) / 10 * 0.3 +
        (df['vibration_x'] - 2.5) / 0.8 * 0.2 +
        (65 - df['pressure']) / 20 * 0.2 +
        (df['operating_hours'] / 8760) * 0.15 +
        (df['maintenance_age'] / 180) * 0.15
    )

    # Convert to failure probability
    failure_probability = 1 / (1 + np.exp(-failure_score))

    # Create binary failure indicator (within next 7 days)
    df['failure_within_days'] = (failure_probability > 0.6).astype(int)

    return df
