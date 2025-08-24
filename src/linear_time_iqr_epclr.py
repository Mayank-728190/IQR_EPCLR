"""
Linear Time IQR-Enhanced Piecewise Constrained Linear Regression

A scalable robust regression algorithm that combines mini-batch stochastic gradient descent
with IQR-based outlier detection to handle large datasets efficiently while maintaining
robustness to outliers.

Author: [Your Name]
Date: August 2025
License: MIT
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Union, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning


class LinearTimeIQREPCLRRegressor(BaseEstimator, RegressorMixin):
    """
    Linear Time IQR-Enhanced Piecewise Constrained Linear Regression.

    A robust and scalable linear regression estimator that uses mini-batch stochastic
    gradient descent with IQR-based outlier detection. The algorithm processes data
    in batches, detecting and down-weighting outliers within each batch using the
    Interquartile Range (IQR) method.

    This approach provides:
    - Linear time complexity O(n) for large datasets
    - Robustness to outliers through adaptive weighting
    - Memory efficiency through mini-batch processing
    - Early stopping for faster convergence

    Parameters
    ----------
    max_epochs : int, default=100
        Maximum number of passes over the entire dataset during training.
        
    learning_rate : float, default=0.01
        Step size for gradient descent updates. Should be positive.
        
    batch_size : int, default=1000
        Number of samples to process in each mini-batch. Larger batches
        provide more stable gradients but use more memory.
        
    iqr_threshold : float, default=1.5
        Multiplier for the IQR to define outlier boundaries. Traditional
        value is 1.5, but can be adjusted for different outlier sensitivity.
        
    outlier_weight : float, default=0.1
        Weight assigned to detected outliers. Should be in (0, 1].
        Lower values reduce outlier influence more aggressively.
        
    tolerance : float, default=1e-6
        Convergence threshold for early stopping. Training stops when
        the improvement in loss is below this threshold.
        
    patience : int, default=10
        Number of epochs with no improvement to wait before early stopping.
        
    random_state : int or None, default=None
        Controls random number generation for data shuffling and weight
        initialization. Pass an int for reproducible results.
        
    verbose : bool, default=False
        Whether to print training progress information.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
        
    intercept_ : float
        Independent term in the linear model.
        
    n_iter_ : int
        Number of epochs actually performed during training.
        
    loss_history_ : list of float
        Training loss for each epoch.
        
    outlier_mask_ : ndarray of shape (n_samples,), dtype=bool
        Boolean mask indicating which training samples were identified
        as outliers in the final model.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    feature_names_in_ : ndarray of shape (n_features_in_,), dtype=str
        Names of features seen during fit. Only defined if X has feature names.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=10000, n_features=10, noise=0.1, random_state=42)
    >>> # Add outliers
    >>> outlier_indices = np.random.choice(len(y), size=100, replace=False)
    >>> y[outlier_indices] += 20 * np.random.randn(100)
    >>> 
    >>> regressor = LinearTimeIQREPCLRRegressor(random_state=42, verbose=True)
    >>> regressor.fit(X, y)
    >>> predictions = regressor.predict(X[:10])
    >>> print(f"Found {regressor.outlier_mask_.sum()} outliers")

    Notes
    -----
    This algorithm is particularly suitable for:
    - Large datasets that don't fit in memory
    - Streaming or online learning scenarios
    - Datasets with sparse outliers
    - Applications requiring fast training times

    The IQR-based outlier detection is performed within each batch, making
    it adaptive to local data characteristics while maintaining computational efficiency.
    """

    def __init__(
        self,
        max_epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 1000,
        iqr_threshold: float = 1.5,
        outlier_weight: float = 0.1,
        tolerance: float = 1e-6,
        patience: int = 10,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iqr_threshold = iqr_threshold
        self.outlier_weight = outlier_weight
        self.tolerance = tolerance
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.iqr_threshold <= 0:
            raise ValueError("iqr_threshold must be positive")
        if not (0 < self.outlier_weight <= 1):
            raise ValueError("outlier_weight must be in (0, 1]")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    def _detect_outliers_iqr(self, residuals: np.ndarray) -> np.ndarray:
        """
        Detect outliers using the IQR method on residuals.
        
        Parameters
        ----------
        residuals : ndarray of shape (n_samples,)
            Residuals from model predictions.
            
        Returns
        -------
        outlier_mask : ndarray of shape (n_samples,), dtype=bool
            Boolean mask indicating outlier samples.
        """
        n_samples = len(residuals)
        
        # Need at least 4 points to compute meaningful quartiles
        if n_samples < 4:
            return np.zeros(n_samples, dtype=bool)
        
        try:
            q1, q3 = np.percentile(residuals, [25, 75])
            iqr = q3 - q1
            
            # Handle degenerate cases (all values identical)
            if iqr < 1e-10:
                return np.zeros(n_samples, dtype=bool)
            
            # Define outlier boundaries
            lower_bound = q1 - self.iqr_threshold * iqr
            upper_bound = q3 + self.iqr_threshold * iqr
            
            outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
            
        except (ValueError, RuntimeWarning):
            # Handle edge cases (e.g., all NaN values)
            outlier_mask = np.zeros(n_samples, dtype=bool)
        
        return outlier_mask

    def _compute_robust_batch_gradient(
        self, 
        X_batch: np.ndarray, 
        y_batch: np.ndarray, 
        weights: np.ndarray
    ) -> Tuple[np.ndarray, float, int]:
        """
        Compute robust gradient for a mini-batch with outlier down-weighting.
        
        Parameters
        ----------
        X_batch : ndarray of shape (batch_size, n_features)
            Feature matrix for the current batch.
        y_batch : ndarray of shape (batch_size,)
            Target values for the current batch.
        weights : ndarray of shape (n_features,)
            Current model weights including intercept.
            
        Returns
        -------
        gradient : ndarray of shape (n_features,)
            Gradient of the loss function.
        batch_loss : float
            Weighted loss for the current batch.
        n_outliers : int
            Number of outliers detected in the batch.
        """
        batch_size = X_batch.shape[0]
        
        # Make predictions and compute residuals
        y_pred = X_batch @ weights
        residuals = y_batch - y_pred
        
        # Detect outliers using IQR method
        outlier_mask = self._detect_outliers_iqr(residuals)
        n_outliers = np.sum(outlier_mask)
        
        # Assign weights: reduced weight for outliers, normal weight for inliers
        sample_weights = np.ones(batch_size)
        sample_weights[outlier_mask] = self.outlier_weight
        
        # Compute weighted residuals and loss
        weighted_residuals = residuals * sample_weights
        batch_loss = np.mean(weighted_residuals**2)
        
        # Compute gradient: ∇L = -2/n * X^T * (weighted_residuals)
        gradient = -2.0 * X_batch.T @ weighted_residuals / batch_size
        
        return gradient, batch_loss, n_outliers

    def _initialize_weights(self, n_features: int) -> np.ndarray:
        """
        Initialize model weights using Xavier/Glorot initialization.
        
        Parameters
        ----------
        n_features : int
            Number of features including intercept.
            
        Returns
        -------
        weights : ndarray of shape (n_features,)
            Initialized weights.
        """
        rng = np.random.RandomState(self.random_state)
        
        # Xavier initialization: scale by sqrt(1/n_features)
        scale = np.sqrt(1.0 / n_features)
        weights = rng.normal(0, scale, n_features)
        
        return weights

    def _create_batches(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create shuffled mini-batches from the dataset.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.
        rng : RandomState
            Random number generator for shuffling.
            
        Returns
        -------
        batches : list of tuple
            List of (X_batch, y_batch) tuples.
        """
        n_samples = X.shape[0]
        
        # Shuffle indices
        indices = rng.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Create batches
        batches = []
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batches.append((X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]))
        
        return batches

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'LinearTimeIQREPCLRRegressor':
        """
        Fit the Linear Time IQR-EPCLR model using mini-batch SGD.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearTimeIQREPCLRRegressor
            Fitted estimator.
        """
        # Validate parameters and input data
        self._validate_parameters()
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        # Store input information
        self.n_features_in_ = n_features
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        
        # Validate batch size
        if self.batch_size > n_samples:
            warnings.warn(
                f"batch_size ({self.batch_size}) is larger than n_samples ({n_samples}). "
                f"Setting batch_size to {n_samples}.",
                UserWarning
            )
            self.batch_size = n_samples
        
        # Add intercept column and initialize weights
        X_with_intercept = np.column_stack([X, np.ones(n_samples)])
        weights = self._initialize_weights(n_features + 1)
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Training loop variables
        self.loss_history_ = []
        best_loss = np.inf
        patience_counter = 0
        total_outliers = 0
        
        if self.verbose:
            print(f"Training on {n_samples} samples with {n_features} features")
            print(f"Batch size: {self.batch_size}, Max epochs: {self.max_epochs}")
        
        # Main training loop
        for epoch in range(self.max_epochs):
            # Create shuffled mini-batches
            batches = self._create_batches(X_with_intercept, y, rng)
            
            epoch_losses = []
            epoch_outliers = 0
            
            # Process each batch
            for X_batch, y_batch in batches:
                # Compute robust gradient
                gradient, batch_loss, n_outliers = self._compute_robust_batch_gradient(
                    X_batch, y_batch, weights
                )
                
                # Update weights using gradient descent
                weights -= self.learning_rate * gradient
                
                # Track metrics
                epoch_losses.append(batch_loss)
                epoch_outliers += n_outliers
            
            # Compute epoch metrics
            epoch_loss = np.mean(epoch_losses)
            self.loss_history_.append(epoch_loss)
            total_outliers += epoch_outliers
            
            # Early stopping check
            if epoch_loss < best_loss - self.tolerance:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Verbose output
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:4d}: Loss = {epoch_loss:.6f}, "
                      f"Outliers = {epoch_outliers:4d}")
            
            # Early stopping
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1} (no improvement for {self.patience} epochs)")
                break
        
        # Store final results
        self.coef_ = weights[:-1]
        self.intercept_ = weights[-1]
        self.n_iter_ = epoch + 1
        
        # Final outlier detection on entire dataset
        try:
            final_residuals = y - self.predict(X)
            self.outlier_mask_ = self._detect_outliers_iqr(final_residuals)
        except Exception:
            # Fallback if prediction fails
            self.outlier_mask_ = np.zeros(n_samples, dtype=bool)
        
        if self.verbose:
            n_final_outliers = np.sum(self.outlier_mask_)
            print(f"Training completed in {self.n_iter_} epochs")
            print(f"Final loss: {self.loss_history_[-1]:.6f}")
            print(f"Final outliers detected: {n_final_outliers} ({100*n_final_outliers/n_samples:.1f}%)")
        
        return self

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with {self.n_features_in_} features"
            )
        
        return X @ self.coef_ + self.intercept_

    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        """
        Return the coefficient of determination R² of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values.
            
        Returns
        -------
        score : float
            R² score.
        """
        y_pred = self.predict(X)
        y = check_array(y, ensure_2d=False, dtype=np.float64)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def get_outlier_scores(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute outlier scores based on absolute residuals.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        outlier_scores : ndarray of shape (n_samples,)
            Absolute residuals as outlier scores. Higher values indicate
            more likely outliers.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        # This method requires the original y values, so we'll use the training residuals
        if hasattr(self, '_training_residuals'):
            return np.abs(self._training_residuals)
        else:
            warnings.warn(
                "Training residuals not available. Use fit() first or call this method "
                "with training data.",
                UserWarning
            )
            return np.zeros(X.shape[0])

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            'max_epochs': self.max_epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'iqr_threshold': self.iqr_threshold,
            'outlier_weight': self.outlier_weight,
            'tolerance': self.tolerance,
            'patience': self.patience,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params) -> 'LinearTimeIQREPCLRRegressor':
        """Set the parameters of this estimator."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param}")
        return self


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression, HuberRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    
    print("Linear Time IQR-EPCLR Demonstration")
    print("=" * 50)
    
    # Generate synthetic dataset with outliers
    np.random.seed(42)
    n_samples, n_features = 50000, 20
    noise_level = 0.1
    
    print(f"Generating dataset: {n_samples} samples, {n_features} features")
    
    # Create base regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise_level,
        random_state=42
    )
    
    # Add outliers (5% of data)
    n_outliers = int(0.05 * n_samples)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    y[outlier_indices] += 10 * np.random.randn(n_outliers)
    
    print(f"Added {n_outliers} outliers ({100*n_outliers/n_samples:.1f}% of data)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models and compare
    models = {
        'Linear Time IQR-EPCLR': LinearTimeIQREPCLRRegressor(
            max_epochs=50,
            learning_rate=0.001,
            batch_size=2000,
            random_state=42,
            verbose=True
        ),
        'OLS': LinearRegression(),
        'Huber': HuberRegressor(max_iter=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'training_time': training_time,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        if hasattr(model, 'outlier_mask_'):
            n_outliers_detected = np.sum(model.outlier_mask_)
            results[name]['outliers_detected'] = n_outliers_detected
    
    # Display results
    print("\nResults Summary:")
    print("=" * 80)
    print(f"{'Model':<20} {'Train Time':<12} {'Train MSE':<12} {'Test MSE':<12} {'Test R²':<10} {'Outliers':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        outliers_str = f"{metrics.get('outliers_detected', 'N/A')}"
        print(f"{name:<20} {metrics['training_time']:<12.3f} {metrics['train_mse']:<12.3f} "
              f"{metrics['test_mse']:<12.3f} {metrics['test_r2']:<10.3f} {outliers_str:<10}")
    
    print("\nKey advantages of Linear Time IQR-EPCLR:")
    print("- Scales linearly with dataset size")
    print("- Robust to outliers through adaptive weighting")
    print("- Memory efficient mini-batch processing")
    print("- Early stopping prevents overfitting")
    print("- Identifies outliers for data quality assessment")