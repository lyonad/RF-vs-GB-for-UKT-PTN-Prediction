"""
Models module for Indonesian Public University Tuition Fees prediction project.
This module implements Random Forest and Gradient-Boosted Trees (XGBoost, CatBoost, LightGBM).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
from typing import Dict, Any, Tuple
import joblib
import os
from sklearn.base import BaseEstimator, RegressorMixin

class MultiOutputQuantileRegressor:
    """
    A wrapper class for quantile regression with multi-output support.
    Implements proper quantile regression using model-specific quantile capabilities.
    """
    
    def __init__(self, base_regressor, quantiles=[0.1, 0.5, 0.9]):
        """
        Initialize the MultiOutputQuantileRegressor.
        
        Args:
            base_regressor: Base regressor to use for quantile regression
            quantiles (list): List of quantiles to predict
        """
        self.base_regressor = base_regressor
        self.quantiles = quantiles
        self.models = {}
    
    def fit(self, X, y):
        """
        Fit the quantile regressor using proper quantile regression approach.
        
        Args:
            X: Training features
            y: Training targets
        """
        from sklearn.multioutput import MultiOutputRegressor
        import xgboost as xgb
        from catboost import CatBoostRegressor
        import lightgbm as lgb
        
        # For each quantile, train a separate model with appropriate quantile settings
        for q in self.quantiles:
            # Clone the base regressor with appropriate parameters for quantile regression
            if hasattr(self.base_regressor, 'get_params'):
                params = self.base_regressor.get_params()
                
                # Remove parameters that might interfere with quantile regression
                clean_params = {k: v for k, v in params.items() 
                              if k not in ['loss', 'objective', 'loss_function', 'alpha', 'quantile_alpha']}
                
                # Set quantile-specific parameters based on the model type
                if isinstance(self.base_regressor, xgb.XGBRegressor):
                    # XGBoost supports quantile regression
                    xgb_params = clean_params.copy()
                    xgb_params['objective'] = 'reg:quantileerror' if q != 0.5 else 'reg:squarederror'
                    if q != 0.5:
                        xgb_params['quantile_alpha'] = q
                    base_model_instance = xgb.XGBRegressor(**xgb_params)
                    
                elif isinstance(self.base_regressor, CatBoostRegressor):
                    # CatBoost supports quantile loss
                    cb_params = clean_params.copy()
                    cb_params['loss_function'] = f'Quantile:alpha={q}' if q != 0.5 else 'RMSE'
                    # Ensure verbose is properly set
                    if 'verbose' in cb_params:
                        cb_params['verbose'] = False
                    base_model_instance = CatBoostRegressor(**cb_params)
                    
                elif isinstance(self.base_regressor, lgb.LGBMRegressor):
                    # LightGBM supports quantile regression
                    lgb_params = clean_params.copy()
                    if q == 0.5:
                        lgb_params['objective'] = 'regression'
                    else:
                        lgb_params['objective'] = 'quantile'
                        lgb_params['alpha'] = q
                    base_model_instance = lgb.LGBMRegressor(**lgb_params)
                    
                elif isinstance(self.base_regressor, RandomForestRegressor):
                    # For RandomForest, we'll use the standard approach
                    # as it doesn't have native quantile regression support
                    base_model_instance = RandomForestRegressor(**clean_params)
                    
                else:
                    # Fallback for other regressors
                    base_model_instance = self.base_regressor.__class__(**clean_params)
            else:
                # For direct regressors without get_params
                base_model_instance = self.base_regressor.__class__()
            
            # Wrap with MultiOutputRegressor for multi-output support
            cloned_model = MultiOutputRegressor(base_model_instance)
            
            # Fit the model
            cloned_model.fit(X, y)
            self.models[q] = cloned_model
        
        return self
    
    def predict(self, X):
        """
        Predict using the quantile regressor.
        
        Args:
            X: Input features
            
        Returns:
            dict: Predictions for each quantile
        """
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X)
        return predictions


class TuitionFeePredictor:
    """
    A class to implement various models for predicting tuition fees.
    """
    
    def __init__(self):
        """
        Initialize the TuitionFeePredictor with all required models.
        """
        self.models = {
            'RandomForest': MultiOutputRegressor(RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )),
            'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )),
            'CatBoost': MultiOutputRegressor(CatBoostRegressor(
                n_estimators=100,
                random_state=42,
                verbose=False
            )),
            'LightGBM': MultiOutputRegressor(lgb.LGBMRegressor(
                n_estimators=100,
                random_state=42,
                verbose=-1
            ))
        }
        
        self.trained_models = {}
        self.quantile_models = {}
    
    def fit_models(self, X_train, y_train):
        """
        Train all models on the training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        
        # Also create quantile models for each base model
        for name, base_model in self.models.items():
            print(f"Training quantile model for {name}...")
            # Extract the underlying estimator from the MultiOutputRegressor
            if hasattr(base_model, 'estimator'):
                # For MultiOutputRegressor, get the underlying estimator
                base_estimator = base_model.estimator
            else:
                # For direct regressors, use as is
                base_estimator = base_model
            
            quantile_model = MultiOutputQuantileRegressor(
                base_estimator,
                quantiles=[0.1, 0.5, 0.9]
            )
            quantile_model.fit(X_train, y_train)
            self.quantile_models[name] = quantile_model
    
    def predict(self, X_test):
        """
        Make predictions using all trained models.
        
        Args:
            X_test: Test features
            
        Returns:
            dict: Predictions from all models
        """
        predictions = {}
        for name, model in self.trained_models.items():
            predictions[name] = model.predict(X_test)
        return predictions
    
    def predict_quantiles(self, X_test):
        """
        Make quantile predictions using all trained models.
        
        Args:
            X_test: Test features
            
        Returns:
            dict: Quantile predictions from all models
        """
        quantile_predictions = {}
        for name, model in self.quantile_models.items():
            quantile_predictions[name] = model.predict(X_test)
        return quantile_predictions
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name in self.trained_models:
            joblib.dump(self.trained_models[model_name], filepath)
            print(f"Model {model_name} saved to {filepath}")
        else:
            print(f"Model {model_name} not found in trained models")
    
    def load_model(self, model_name, filepath):
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name of the model to load
            filepath (str): Path to load the model from
        """
        if os.path.exists(filepath):
            loaded_model = joblib.load(filepath)
            self.trained_models[model_name] = loaded_model
            print(f"Model {model_name} loaded from {filepath}")
        else:
            print(f"File {filepath} does not exist")