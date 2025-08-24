"""
IQR-EPCLR: Interquartile Range-Efficient Projection Candidate Linear Regression

A robust regression estimator with high breakdown point and computational efficiency.
This implementation follows the three-stage MM-type estimation procedure described in
the literature for robust regression with deterministic candidate selection.

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


class IQREPCLRRegressor(BaseEstimator, RegressorMixin):
    """
    IQR-EPCLR: A Scalable and Deterministic High-Breakdown Regression Estimator.

    This robust regression estimator implements a three-stage MM-type estimation procedure:
    
    1. **Initial Estimation**: A deterministic initial estimate with 25% breakdown point
       is obtained by minimizing the IQR of absolute residuals over a set of 
       deterministic candidate subsets.
    
    2. **Scale Estimation**: A high-breakdown (50%) scale estimate is computed using
       the Median Absolute Deviation (MAD) of residuals from the initial fit.
    
    3. **Final Estimation**: An efficient M-estimate is computed using Iteratively
       Reweighted Least Squares (IRLS) with a Tukey biweight function.

    Parameters
    ----------
    n_candidates : int, default=500
        Number of deterministic candidate subsets to generate for initial estimation.
        Higher values improve robustness but increase computational cost.
        
    max_iterations : int, default=50
        Maximum number of IRLS iterations for final M-estimation.
        
    c_tukey : float, default=4.685
        Tuning constant for the Tukey biweight function. The default value
        corresponds to 95% efficiency at the normal distribution.
        
    tolerance : float, default=1e-6
        Convergence tolerance for IRLS iterations. Algorithm stops when
        the relative change in coefficients is below this threshold.
        
    random_state : int or None, default=None
        Controls the random number generation for candidate subset selection.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
        
    intercept_ : float
        Independent term in the linear model.
        
    scale_ : float
        Robust scale estimate of the residuals.
        
    n_iter_ : int
        Number of IRLS iterations performed during fitting.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    feature_names_in_ : ndarray of shape (n_features_in_,), dtype=str
        Names of features seen during fit. Only defined if X has feature names.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    >>> # Add some outliers
    >>> y[0:5] += 10
    >>> regressor = IQREPCLRRegressor(random_state=42)
    >>> regressor.fit(X, y)
    IQREPCLRRegressor(random_state=42)
    >>> predictions = regressor.predict(X[:5])

    References
    ----------
    [1] "IQR-EPCLR: A Scalable and Deterministic High-Breakdown Regression Estimator"
    [2] Rousseeuw, P.J. and Yohai, V.J. (1984). "Robust regression by means of 
        S-estimators." In Robust and Nonlinear Time Series Analysis.
    """

    def __init__(
        self,
        n_candidates: int = 500,
        max_iterations: int = 50,
        c_tukey: float = 4.685,
        tolerance: float = 1e-6,
        random_state: Optional[int] = None
    ):
        self.n_candidates = n_candidates
        self.max_iterations = max_iterations
        self.c_tukey = c_tukey
        self.tolerance = tolerance
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self.n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.c_tukey <= 0:
            raise ValueError("c_tukey must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")

    def _compute_leverage_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute leverage scores (diagonal elements of hat matrix) safely.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input matrix including intercept column.
            
        Returns
        -------
        leverage : ndarray of shape (n_samples,)
            Leverage scores for each observation.
        """
        try:
            # QR decomposition is more numerically stable
            Q, _ = np.linalg.qr(X, mode='reduced')
            leverage = np.sum(Q**2, axis=1)
        except np.linalg.LinAlgError:
            # Fallback for highly singular matrices
            warnings.warn(
                "Could not compute leverage scores reliably. Using uniform weights.",
                ConvergenceWarning
            )
            leverage = np.ones(X.shape[0]) / X.shape[0]
        
        return leverage

    def _generate_candidate_subsets(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Generate deterministic candidate subsets for initial estimation.
        
        This method uses a multi-strategy approach to ensure good coverage
        of the parameter space while maintaining deterministic behavior.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix including intercept column.
            
        Returns
        -------
        subsets : list of ndarray
            List of candidate subset indices.
        """
        n_samples, n_features = X.shape
        
        if n_samples < n_features:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= number of features ({n_features})"
            )

        # Set random state for reproducibility
        rng = np.random.RandomState(self.random_state)
        
        # Compute data-driven selection criteria
        leverage_scores = self._compute_leverage_scores(X)
        composite_feature = np.sum(X[:, 1:], axis=1)  # Exclude intercept column
        
        # Get sorted indices
        leverage_indices = np.argsort(leverage_scores)
        composite_indices = np.argsort(composite_feature)
        
        subsets = []
        subsets_per_strategy = max(1, self.n_candidates // 4)
        
        # Strategy 1: High leverage observations
        high_leverage_pool = leverage_indices[-max(n_features * 2, n_samples // 4):]
        for _ in range(subsets_per_strategy):
            if len(high_leverage_pool) >= n_features:
                subset = rng.choice(high_leverage_pool, n_features, replace=False)
                subsets.append(subset)
        
        # Strategy 2: Low leverage observations
        low_leverage_pool = leverage_indices[:max(n_features * 2, n_samples // 4)]
        for _ in range(subsets_per_strategy):
            if len(low_leverage_pool) >= n_features:
                subset = rng.choice(low_leverage_pool, n_features, replace=False)
                subsets.append(subset)
        
        # Strategy 3: Extreme composite feature values
        n_extreme = max(n_features * 3, n_samples // 6)
        extreme_indices = np.concatenate([
            composite_indices[:n_extreme],
            composite_indices[-n_extreme:]
        ])
        for _ in range(subsets_per_strategy):
            if len(extreme_indices) >= n_features:
                subset = rng.choice(extreme_indices, n_features, replace=False)
                subsets.append(subset)
        
        # Strategy 4: Purely random subsets for diversity
        while len(subsets) < self.n_candidates:
            subset = rng.choice(n_samples, n_features, replace=False)
            subsets.append(subset)
        
        return subsets

    def _initial_estimate(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Stage 1: Compute initial robust estimate by minimizing IQR of residuals.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix including intercept.
        y : ndarray of shape (n_samples,)
            Target values.
            
        Returns
        -------
        beta_initial : ndarray of shape (n_features,)
            Initial coefficient estimates.
        best_iqr : float
            IQR of residuals for the best initial estimate.
        """
        subsets = self._generate_candidate_subsets(X)
        
        best_iqr = np.inf
        beta_initial = None
        n_successful = 0
        
        for subset_indices in subsets:
            try:
                X_subset = X[subset_indices]
                y_subset = y[subset_indices]
                
                # Check condition number to avoid nearly singular systems
                if np.linalg.cond(X_subset) > 1e12:
                    continue
                
                # Solve the exact system
                beta_candidate = np.linalg.solve(X_subset, y_subset)
                
                # Compute residuals and IQR
                residuals = np.abs(y - X @ beta_candidate)
                q75, q25 = np.percentile(residuals, [75, 25])
                iqr = q75 - q25
                
                if iqr < best_iqr:
                    best_iqr = iqr
                    beta_initial = beta_candidate
                    n_successful += 1
                    
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        if beta_initial is None:
            raise RuntimeError(
                "Could not find any non-singular subset for initial estimation. "
                "Try increasing n_candidates or check your data for collinearity."
            )
        
        if n_successful < self.n_candidates * 0.1:
            warnings.warn(
                f"Only {n_successful}/{self.n_candidates} candidate subsets were successful. "
                "Consider checking for numerical issues in your data.",
                ConvergenceWarning
            )
        
        return beta_initial, best_iqr

    def _scale_estimate(self, residuals: np.ndarray) -> float:
        """
        Stage 2: Compute high-breakdown scale estimate using MAD.
        
        Parameters
        ----------
        residuals : ndarray of shape (n_samples,)
            Residuals from initial fit.
            
        Returns
        -------
        scale : float
            Robust scale estimate.
        """
        # Median Absolute Deviation with consistency factor for normal distribution
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = mad / 0.6745  # Asymptotic consistency factor
        
        # Prevent numerical issues with perfect or near-perfect fits
        return max(scale, 1e-10)

    def _tukey_biweight_weights(self, scaled_residuals: np.ndarray) -> np.ndarray:
        """
        Compute Tukey biweight weights for IRLS.
        
        Parameters
        ----------
        scaled_residuals : ndarray of shape (n_samples,)
            Residuals scaled by robust scale estimate.
            
        Returns
        -------
        weights : ndarray of shape (n_samples,)
            Tukey biweight weights.
        """
        abs_scaled = np.abs(scaled_residuals)
        weights = np.zeros_like(scaled_residuals)
        
        # Only assign non-zero weights to observations within the threshold
        mask = abs_scaled <= self.c_tukey
        if np.any(mask):
            u = scaled_residuals[mask] / self.c_tukey
            weights[mask] = (1 - u**2)**2
        
        return weights

    def _m_estimate(self, X: np.ndarray, y: np.ndarray, beta_init: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Stage 3: Final efficient M-estimation using IRLS with Tukey biweight.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix including intercept.
        y : ndarray of shape (n_samples,)
            Target values.
        beta_init : ndarray of shape (n_features,)
            Initial coefficient estimates.
            
        Returns
        -------
        beta_final : ndarray of shape (n_features,)
            Final coefficient estimates.
        n_iter : int
            Number of iterations performed.
        """
        beta_current = beta_init.copy()
        
        for iteration in range(self.max_iterations):
            # Compute residuals and weights
            residuals = y - X @ beta_current
            scaled_residuals = residuals / self.scale_
            weights = self._tukey_biweight_weights(scaled_residuals)
            
            # Check if all weights are zero (complete breakdown)
            if np.sum(weights) < 1e-10:
                warnings.warn(
                    "All observations have zero weight. Returning initial estimate.",
                    ConvergenceWarning
                )
                return beta_init, iteration + 1
            
            # Weighted least squares
            sqrt_weights = np.sqrt(weights)
            X_weighted = X * sqrt_weights[:, np.newaxis]
            y_weighted = y * sqrt_weights
            
            try:
                # Use pseudo-inverse for numerical stability
                XtX = X_weighted.T @ X_weighted
                Xty = X_weighted.T @ y_weighted
                
                # Add small regularization if needed
                if np.linalg.cond(XtX) > 1e12:
                    XtX += np.eye(XtX.shape[0]) * 1e-10
                
                beta_new = np.linalg.solve(XtX, Xty)
                
            except (np.linalg.LinAlgError, ValueError):
                warnings.warn(
                    "Numerical issues in IRLS. Returning current estimate.",
                    ConvergenceWarning
                )
                return beta_current, iteration + 1
            
            # Check convergence
            if np.linalg.norm(beta_new - beta_current) < self.tolerance * (1 + np.linalg.norm(beta_current)):
                return beta_new, iteration + 1
            
            beta_current = beta_new
        
        warnings.warn(
            f"IRLS did not converge after {self.max_iterations} iterations.",
            ConvergenceWarning
        )
        return beta_current, self.max_iterations

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'IQREPCLRRegressor':
        """
        Fit the IQR-EPCLR robust regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : IQREPCLRRegressor
            Fitted estimator.
        """
        # Validate parameters and input data
        self._validate_parameters()
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store input information
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        try:
            # Stage 1: Initial robust estimate
            beta_initial, _ = self._initial_estimate(X_with_intercept, y)
            
            # Stage 2: Scale estimation
            initial_residuals = y - X_with_intercept @ beta_initial
            self.scale_ = self._scale_estimate(initial_residuals)
            
            # Stage 3: Final M-estimation
            beta_final, n_iter = self._m_estimate(X_with_intercept, y, beta_initial)
            
            # Store results
            self.intercept_ = beta_final[0]
            self.coef_ = beta_final[1:]
            self.n_iter_ = n_iter
            
        except Exception as e:
            raise RuntimeError(f"Fitting failed: {str(e)}")
        
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
        X = check_array(X, accept_sparse=False)
        
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
        y = check_array(y, ensure_2d=False)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            'n_candidates': self.n_candidates,
            'max_iterations': self.max_iterations,
            'c_tukey': self.c_tukey,
            'tolerance': self.tolerance,
            'random_state': self.random_state
        }

    def set_params(self, **params) -> 'IQREPCLRRegressor':
        """Set the parameters of this estimator."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param}")
        return self


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data with outliers for demonstration
    np.random.seed(42)
    n_samples, n_features = 100, 3
    
    # Generate clean data
    X_clean = np.random.randn(n_samples, n_features)
    true_coef = np.array([1.5, -2.0, 0.5])
    y_clean = X_clean @ true_coef + 0.1 * np.random.randn(n_samples)
    
    # Add outliers
    n_outliers = 10
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    y_contaminated = y_clean.copy()
    y_contaminated[outlier_indices] += 5 * np.random.randn(n_outliers)
    
    # Fit the robust model
    robust_reg = IQREPCLRRegressor(n_candidates=200, random_state=42)
    robust_reg.fit(X_clean, y_contaminated)
    
    print("True coefficients:", true_coef)
    print("Estimated coefficients:", robust_reg.coef_)
    print("Scale estimate:", robust_reg.scale_)
    print("Number of IRLS iterations:", robust_reg.n_iter_)
    
    # Compare with ordinary least squares
    from sklearn.linear_model import LinearRegression
    ols_reg = LinearRegression()
    ols_reg.fit(X_clean, y_contaminated)
    
    print("\nOLS coefficients (contaminated):", ols_reg.coef_)
    print("Robust R² score:", robust_reg.score(X_clean, y_clean))
    print("OLS R² score:", ols_reg.score(X_clean, y_clean))