"""
Ridge Regression Decoder
========================

Ridge regression decoder for continuous velocity prediction from neural data.
Includes training, evaluation, and visualization tools.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

class RidgeVelocityDecoder:
    """
    Ridge regression decoder for velocity prediction from neural data.
    
    Implements Ridge regression with cross-validation, hyperparameter tuning,
    and comprehensive evaluation metrics.
    """
    
    def __init__(self, alpha: float = 1.0, 
                 normalize_features: bool = True,
                 normalize_targets: bool = False):
        """
        Initialize Ridge decoder.
        
        Parameters:
        -----------
        alpha : float
            Ridge regression regularization parameter
        normalize_features : bool
            Whether to normalize neural features
        normalize_targets : bool
            Whether to normalize velocity targets
        """
        self.alpha = alpha
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        
        # Initialize models and scalers
        self.model_x = Ridge(alpha=alpha)
        self.model_y = Ridge(alpha=alpha)
        
        if normalize_features:
            self.feature_scaler = StandardScaler()
        if normalize_targets:
            self.target_scaler = StandardScaler()
        
        # Training history
        self.is_trained = False
        self.training_history = {}
        
        print(f"🧠 RidgeVelocityDecoder initialized:")
        print(f"  • Alpha: {self.alpha}")
        print(f"  • Normalize features: {self.normalize_features}")
        print(f"  • Normalize targets: {self.normalize_targets}")
    
    def prepare_data(self, neural_features: np.ndarray, 
                    behavioral_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training/prediction.
        
        Parameters:
        -----------
        neural_features : np.ndarray
            Neural features (n_samples x n_channels)
        behavioral_targets : np.ndarray
            Behavioral targets (n_samples x 2) for [velocity_x, velocity_y]
            
        Returns:
        --------
        tuple
            (prepared_features, prepared_targets)
        """
        X = neural_features.copy()
        y = behavioral_targets.copy()
        
        # Normalize features
        if self.normalize_features:
            if hasattr(self, 'feature_scaler') and hasattr(self.feature_scaler, 'mean_'):
                # Use existing scaler (for prediction)
                X = self.feature_scaler.transform(X)
            else:
                # Fit new scaler (for training)
                X = self.feature_scaler.fit_transform(X)
        
        # Normalize targets
        if self.normalize_targets:
            if hasattr(self, 'target_scaler') and hasattr(self.target_scaler, 'mean_'):
                # Use existing scaler (for prediction)
                y = self.target_scaler.transform(y)
            else:
                # Fit new scaler (for training)
                y = self.target_scaler.fit_transform(y)
        
        return X, y
    
    def train(self, neural_features: np.ndarray, 
              behavioral_targets: np.ndarray,
              test_size: float = 0.2,
              random_state: int = 42) -> Dict:
        """
        Train Ridge regression models.
        
        Parameters:
        -----------
        neural_features : np.ndarray
            Neural features (n_samples x n_channels)
        behavioral_targets : np.ndarray
            Behavioral targets (n_samples x 2)
        test_size : float
            Fraction of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Training results and metrics
        """
        print(f"🏋️ Training Ridge decoder...")
        print(f"  • Training data: {neural_features.shape}")
        print(f"  • Test split: {test_size}")
        
        # Prepare data
        X, y = self.prepare_data(neural_features, behavioral_targets)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        # Train models for each velocity component
        print("  • Training velocity_x model...")
        self.model_x.fit(X_train, y_train[:, 0])
        
        print("  • Training velocity_y model...")
        self.model_y.fit(X_train, y_train[:, 1])
        
        # Evaluate on test set
        y_pred_x = self.model_x.predict(X_test)
        y_pred_y = self.model_y.predict(X_test)
        y_pred = np.column_stack([y_pred_x, y_pred_y])
        
        # Calculate metrics
        training_results = self._calculate_metrics(y_test, y_pred)
        training_results.update({
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'n_channels': X_train.shape[1],
            'alpha': self.alpha,
            'test_data': (X_test, y_test, y_pred)
        })
        
        # Store training history
        self.training_history = training_results
        self.is_trained = True
        
        print(f"✅ Training complete:")
        print(f"  • R² velocity_x: {training_results['r2_x']:.3f}")
        print(f"  • R² velocity_y: {training_results['r2_y']:.3f}")
        print(f"  • Overall correlation: {training_results['overall_correlation']:.3f}")
        
        return training_results
    
    def predict(self, neural_features: np.ndarray) -> np.ndarray:
        """
        Predict velocity from neural features.
        
        Parameters:
        -----------
        neural_features : np.ndarray
            Neural features (n_samples x n_channels)
            
        Returns:
        --------
        np.ndarray
            Predicted velocities (n_samples x 2)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        X, _ = self.prepare_data(neural_features, np.zeros((neural_features.shape[0], 2)))
        
        # Predict
        y_pred_x = self.model_x.predict(X)
        y_pred_y = self.model_y.predict(X)
        y_pred = np.column_stack([y_pred_x, y_pred_y])
        
        # Denormalize if needed
        if self.normalize_targets:
            y_pred = self.target_scaler.inverse_transform(y_pred)
        
        return y_pred
    
    def cross_validate(self, neural_features: np.ndarray, 
                      behavioral_targets: np.ndarray,
                      cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        neural_features : np.ndarray
            Neural features
        behavioral_targets : np.ndarray
            Behavioral targets
        cv_folds : int
            Number of CV folds
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        print(f"🔄 Performing {cv_folds}-fold cross-validation...")
        
        # Prepare data
        X, y = self.prepare_data(neural_features, behavioral_targets)
        
        # Cross-validate each velocity component
        cv_scores_x = cross_val_score(self.model_x, X, y[:, 0], cv=cv_folds, scoring='r2')
        cv_scores_y = cross_val_score(self.model_y, X, y[:, 1], cv=cv_folds, scoring='r2')
        
        cv_results = {
            'cv_scores_x': cv_scores_x,
            'cv_scores_y': cv_scores_y,
            'mean_r2_x': np.mean(cv_scores_x),
            'std_r2_x': np.std(cv_scores_x),
            'mean_r2_y': np.mean(cv_scores_y),
            'std_r2_y': np.std(cv_scores_y),
            'overall_mean_r2': np.mean([np.mean(cv_scores_x), np.mean(cv_scores_y)])
        }
        
        print(f"✅ Cross-validation complete:")
        print(f"  • R² velocity_x: {cv_results['mean_r2_x']:.3f} ± {cv_results['std_r2_x']:.3f}")
        print(f"  • R² velocity_y: {cv_results['mean_r2_y']:.3f} ± {cv_results['std_r2_y']:.3f}")
        
        return cv_results
    
    def hyperparameter_search(self, neural_features: np.ndarray, 
                            behavioral_targets: np.ndarray,
                            alpha_range: List[float] = None,
                            cv_folds: int = 5) -> Dict:
        """
        Search for optimal hyperparameters.
        
        Parameters:
        -----------
        neural_features : np.ndarray
            Neural features
        behavioral_targets : np.ndarray
            Behavioral targets
        alpha_range : list
            Range of alpha values to test
        cv_folds : int
            Number of CV folds
            
        Returns:
        --------
        dict
            Hyperparameter search results
        """
        if alpha_range is None:
            alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        print(f"🔍 Hyperparameter search over {len(alpha_range)} alpha values...")
        
        # Prepare data
        X, y = self.prepare_data(neural_features, behavioral_targets)
        
        results = []
        
        for alpha in alpha_range:
            # Test current alpha
            model_x = Ridge(alpha=alpha)
            model_y = Ridge(alpha=alpha)
            
            # Cross-validate
            cv_scores_x = cross_val_score(model_x, X, y[:, 0], cv=cv_folds, scoring='r2')
            cv_scores_y = cross_val_score(model_y, X, y[:, 1], cv=cv_folds, scoring='r2')
            
            mean_r2 = np.mean([np.mean(cv_scores_x), np.mean(cv_scores_y)])
            
            results.append({
                'alpha': alpha,
                'mean_r2_x': np.mean(cv_scores_x),
                'mean_r2_y': np.mean(cv_scores_y),
                'overall_mean_r2': mean_r2,
                'cv_scores_x': cv_scores_x,
                'cv_scores_y': cv_scores_y
            })
            
            print(f"  • Alpha {alpha:7.3f}: R² = {mean_r2:.3f}")
        
        # Find best alpha
        best_result = max(results, key=lambda x: x['overall_mean_r2'])
        best_alpha = best_result['alpha']
        
        print(f"✅ Best alpha: {best_alpha} (R² = {best_result['overall_mean_r2']:.3f})")
        
        return {
            'results': results,
            'best_alpha': best_alpha,
            'best_result': best_result
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # R² scores
        metrics['r2_x'] = r2_score(y_true[:, 0], y_pred[:, 0])
        metrics['r2_y'] = r2_score(y_true[:, 1], y_pred[:, 1])
        
        # MSE
        metrics['mse_x'] = mean_squared_error(y_true[:, 0], y_pred[:, 0])
        metrics['mse_y'] = mean_squared_error(y_true[:, 1], y_pred[:, 1])
        
        # Correlations
        metrics['correlation_x'], metrics['p_value_x'] = pearsonr(y_true[:, 0], y_pred[:, 0])
        metrics['correlation_y'], metrics['p_value_y'] = pearsonr(y_true[:, 1], y_pred[:, 1])
        
        # Overall correlation (magnitude)
        true_magnitude = np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2)
        pred_magnitude = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)
        metrics['overall_correlation'], metrics['overall_p_value'] = pearsonr(true_magnitude, pred_magnitude)
        
        # Direction correlation
        true_direction = np.arctan2(y_true[:, 1], y_true[:, 0])
        pred_direction = np.arctan2(y_pred[:, 1], y_pred[:, 0])
        # Handle circular correlation
        direction_diff = np.abs(np.angle(np.exp(1j * (true_direction - pred_direction))))
        metrics['direction_error_mean'] = np.mean(direction_diff)
        metrics['direction_error_std'] = np.std(direction_diff)
        
        return metrics
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Velocity Predictions") -> plt.Figure:
        """
        Plot prediction results.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True velocities
        y_pred : np.ndarray
            Predicted velocities
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Figure with prediction plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Velocity X scatter
        axes[0, 0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
        axes[0, 0].plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                       [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Velocity X')
        axes[0, 0].set_ylabel('Predicted Velocity X')
        axes[0, 0].set_title('Velocity X Predictions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Velocity Y scatter
        axes[0, 1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5)
        axes[0, 1].plot([y_true[:, 1].min(), y_true[:, 1].max()], 
                       [y_true[:, 1].min(), y_true[:, 1].max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('True Velocity Y')
        axes[0, 1].set_ylabel('Predicted Velocity Y')
        axes[0, 1].set_title('Velocity Y Predictions')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series (first 500 samples)
        n_samples = min(500, len(y_true))
        time_axis = np.arange(n_samples)
        
        axes[1, 0].plot(time_axis, y_true[:n_samples, 0], 'b-', label='True', alpha=0.7)
        axes[1, 0].plot(time_axis, y_pred[:n_samples, 0], 'r-', label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Time (bins)')
        axes[1, 0].set_ylabel('Velocity X')
        axes[1, 0].set_title('Velocity X Time Series')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time_axis, y_true[:n_samples, 1], 'b-', label='True', alpha=0.7)
        axes[1, 1].plot(time_axis, y_pred[:n_samples, 1], 'r-', label='Predicted', alpha=0.7)
        axes[1, 1].set_xlabel('Time (bins)')
        axes[1, 1].set_ylabel('Velocity Y')
        axes[1, 1].set_title('Velocity Y Time Series')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, channel_indices: List[int] = None) -> plt.Figure:
        """
        Plot feature importance (regression coefficients).
        
        Parameters:
        -----------
        channel_indices : list
            Channel indices for labeling
            
        Returns:
        --------
        plt.Figure
            Feature importance plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting feature importance")
        
        coefs_x = self.model_x.coef_
        coefs_y = self.model_y.coef_
        
        if channel_indices is None:
            channel_indices = list(range(len(coefs_x)))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Velocity X coefficients
        axes[0].bar(range(len(coefs_x)), coefs_x, alpha=0.7)
        axes[0].set_xlabel('Channel Index')
        axes[0].set_ylabel('Coefficient Value')
        axes[0].set_title('Velocity X - Feature Importance')
        axes[0].grid(True, alpha=0.3)
        
        # Velocity Y coefficients
        axes[1].bar(range(len(coefs_y)), coefs_y, alpha=0.7)
        axes[1].set_xlabel('Channel Index')
        axes[1].set_ylabel('Coefficient Value')
        axes[1].set_title('Velocity Y - Feature Importance')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> str:
        """
        Generate a text report of decoder performance.
        
        Returns:
        --------
        str
            Performance report
        """
        if not self.is_trained:
            return "Model not trained yet."
        
        metrics = self.training_history
        
        report = f"""
Ridge Velocity Decoder Performance Report
========================================

Model Configuration:
  • Alpha (regularization): {self.alpha}
  • Feature normalization: {self.normalize_features}
  • Target normalization: {self.normalize_targets}
  • Training samples: {metrics['train_samples']}
  • Test samples: {metrics['test_samples']}
  • Neural channels: {metrics['n_channels']}

Performance Metrics:
  • R² Velocity X: {metrics['r2_x']:.3f}
  • R² Velocity Y: {metrics['r2_y']:.3f}
  • MSE Velocity X: {metrics['mse_x']:.3f}
  • MSE Velocity Y: {metrics['mse_y']:.3f}
  • Correlation Velocity X: {metrics['correlation_x']:.3f} (p={metrics['p_value_x']:.3e})
  • Correlation Velocity Y: {metrics['correlation_y']:.3f} (p={metrics['p_value_y']:.3e})
  • Overall Correlation: {metrics['overall_correlation']:.3f} (p={metrics['overall_p_value']:.3e})

Direction Decoding:
  • Direction Error (mean): {metrics['direction_error_mean']:.3f} rad
  • Direction Error (std): {metrics['direction_error_std']:.3f} rad
        """
        
        return report.strip() 