"""
Conformal Prediction module for Indonesian Public University Tuition Fees prediction project.
This module implements conformal prediction methods for uncertainty quantification.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ConformalPredictor:
    """
    A class to implement conformal prediction for regression tasks.
    """
    
    def __init__(self, model, significance_level=0.1):
        """
        Initialize the ConformalPredictor.
        
        Args:
            model: Base regression model
            significance_level (float): Significance level (alpha) for confidence intervals
        """
        self.model = model
        self.significance_level = significance_level
        self.calibration_errors = None
        self.is_fitted = False
    
    def fit(self, X, y, X_cal=None, y_cal=None, calibration_fraction: float = 0.2, random_state: int = 42):
        """
        Fit the base model and prepare for conformal prediction.
        
        Args:
            X: Training features
            y: Training targets
        """
        # If explicit calibration split provided, use it; otherwise create a deterministic split
        if X_cal is None or y_cal is None:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y, test_size=calibration_fraction, random_state=random_state
            )
        else:
            X_train, y_train = X, y
        
        # Fit the base model on training subset
        self.model.fit(X_train, y_train)
        
        # Get predictions on calibration set
        y_pred_cal = self.model.predict(X_cal)
        
        # Calculate non-conformity scores (absolute residuals)
        self.calibration_errors = np.abs(y_cal - y_pred_cal)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Input features
            
        Returns:
            tuple: (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get point predictions
        y_pred = self.model.predict(X)
        
        # Calculate the quantile of errors for the desired confidence level
        # Handle case where calibration_errors might be 1D or 2D
        if len(self.calibration_errors.shape) > 1:
            error_quantile = np.quantile(self.calibration_errors, 
                                        1 - self.significance_level, axis=0)
        else:
            error_quantile = np.quantile(self.calibration_errors, 
                                        1 - self.significance_level)
        
        # Create prediction intervals
        lower_bounds = y_pred - error_quantile
        upper_bounds = y_pred + error_quantile
        
        return y_pred, lower_bounds, upper_bounds


class MultiOutputConformalPredictor:
    """
    A class to implement conformal prediction for multi-output regression tasks.
    """
    
    def __init__(self, base_model, significance_level=0.1):
        """
        Initialize the MultiOutputConformalPredictor.
        
        Args:
            base_model: Base multi-output regression model
            significance_level (float): Significance level (alpha) for confidence intervals
        """
        self.base_model = base_model
        self.significance_level = significance_level
        self.conformal_predictors = []
        self.is_fitted = False
    
    def fit(self, X, y, calibration_fraction: float = 0.2, random_state: int = 42):
        """
        Fit the conformal predictors for each output.
        
        Args:
            X: Training features
            y: Training targets (multi-output)
        """
        # Convert to numpy array if it's a DataFrame
        if hasattr(y, 'values'):
            y_values = y.values
        else:
            y_values = np.array(y)
        
        n_outputs = y_values.shape[1] if len(y_values.shape) > 1 else 1
        
        # Create a single, shared calibration split to ensure fairness across outputs
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y_values, test_size=calibration_fraction, random_state=random_state
        )
        
        # Handle single output case
        if n_outputs == 1:
            # For single output, create one conformal predictor
            single_model = self._create_single_output_model(0)
            cp = ConformalPredictor(single_model, self.significance_level)
            # Reshape to 1D if needed and fit with shared calibration split
            y_tr_1d = y_tr.ravel() if len(y_tr.shape) > 1 else y_tr
            y_cal_1d = y_cal.ravel() if len(y_cal.shape) > 1 else y_cal
            cp.fit(X_tr, y_tr_1d, X_cal=X_cal, y_cal=y_cal_1d, calibration_fraction=calibration_fraction, random_state=random_state)
            self.conformal_predictors.append(cp)
        else:
            # Create a conformal predictor for each output
            for i in range(n_outputs):
                # Create a single-output model for the i-th output
                single_output_model = self._create_single_output_model(i)
                
                # Create conformal predictor for this output
                cp = ConformalPredictor(single_output_model, self.significance_level)
                
                # Fit on the i-th output
                cp.fit(
                    X_tr,
                    y_tr[:, i],
                    X_cal=X_cal,
                    y_cal=y_cal[:, i],
                    calibration_fraction=calibration_fraction,
                    random_state=random_state,
                )
                
                self.conformal_predictors.append(cp)
        
        self.is_fitted = True
        return self
    
    def _create_single_output_model(self, output_idx):
        """
        Create a single-output version of the base model for a specific output.
        
        Args:
            output_idx (int): Index of the output to predict
            
        Returns:
            A single-output model based on the base model
        """
        # Extract the actual model from the MultiOutputRegressor wrapper
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        from catboost import CatBoostRegressor
        import lightgbm as lgb
        
        # Check if the base model is a MultiOutputRegressor
        if isinstance(self.base_model, MultiOutputRegressor):
            # Get the underlying estimator
            base_estimator = self.base_model.estimator
            
            # Clone the base estimator with the same parameters
            if isinstance(base_estimator, RandomForestRegressor):
                params = base_estimator.get_params()
                single_model = RandomForestRegressor(**{k: v for k, v in params.items() 
                                                       if k in RandomForestRegressor().get_params()})
            elif isinstance(base_estimator, xgb.XGBRegressor):
                params = base_estimator.get_params()
                # Filter out invalid parameters for XGBRegressor
                valid_params = {k: v for k, v in params.items() 
                               if k in xgb.XGBRegressor().get_params()}
                single_model = xgb.XGBRegressor(**valid_params)
            elif isinstance(base_estimator, CatBoostRegressor):
                params = base_estimator.get_params()
                # Remove parameters that are not valid for single output model
                valid_params = {k: v for k, v in params.items() 
                               if k in CatBoostRegressor().get_params() and k not in ['verbose', 'logging_level']}
                single_model = CatBoostRegressor(**valid_params, verbose=False)
            elif isinstance(base_estimator, lgb.LGBMRegressor):
                params = base_estimator.get_params()
                # Filter out invalid parameters for LGBMRegressor
                valid_params = {k: v for k, v in params.items() 
                               if k in lgb.LGBMRegressor().get_params() and k not in ['verbose']}
                single_model = lgb.LGBMRegressor(**valid_params, verbose=-1)
            else:
                # Generic fallback - clone the estimator with same parameters
                if hasattr(base_estimator, 'get_params'):
                    params = base_estimator.get_params()
                    single_model = base_estimator.__class__(**params)
                else:
                    # Fallback: create a new instance
                    single_model = base_estimator.__class__()
        else:
            # If base model is not MultiOutputRegressor, return a clone
            if hasattr(self.base_model, 'get_params'):
                params = self.base_model.get_params()
                single_model = self.base_model.__class__(**params)
            else:
                # Fallback: create a new instance
                single_model = self.base_model.__class__()
        
        return single_model
    
    def predict(self, X):
        """
        Make predictions with confidence intervals for all outputs.
        
        Args:
            X: Input features
            
        Returns:
            tuple: (predictions, lower_bounds, upper_bounds) for all outputs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        all_predictions = []
        all_lower_bounds = []
        all_upper_bounds = []
        
        for cp in self.conformal_predictors:
            pred, lower, upper = cp.predict(X)
            all_predictions.append(pred)
            all_lower_bounds.append(lower)
            all_upper_bounds.append(upper)
        
        # Stack the results
        y_pred = np.column_stack(all_predictions)
        lower_bounds = np.column_stack(all_lower_bounds)
        upper_bounds = np.column_stack(all_upper_bounds)
        
        return y_pred, lower_bounds, upper_bounds


def create_conformal_predictors(models_dict, significance_level=0.1):
    """
    Create conformal predictors for a dictionary of models.
    
    Args:
        models_dict (dict): Dictionary of trained models
        significance_level (float): Significance level for confidence intervals
        
    Returns:
        dict: Dictionary of conformal predictors
    """
    conformal_predictors = {}
    
    for name, model in models_dict.items():
        # Create a conformal predictor for each model
        cp = MultiOutputConformalPredictor(
            model, 
            significance_level=significance_level
        )
        conformal_predictors[name] = cp
    
    return conformal_predictors


def save_conformal_predictors(conformal_predictors, filepath="results/conformal_predictors.pkl"):
    """
    Save conformal predictors to a file.
    
    Args:
        conformal_predictors (dict): Dictionary of conformal predictors
        filepath (str): Path to save the conformal predictors
    """
    import joblib
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(conformal_predictors, filepath)
    print(f"Conformal predictors saved to {filepath}")


def load_conformal_predictors(filepath="results/conformal_predictors.pkl"):
    """
    Load conformal predictors from a file.
    
    Args:
        filepath (str): Path to load the conformal predictors from
        
    Returns:
        dict: Dictionary of conformal predictors
    """
    import joblib
    conformal_predictors = joblib.load(filepath)
    print(f"Conformal predictors loaded from {filepath}")
    return conformal_predictors