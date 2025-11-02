"""
Test script to verify fairness in model comparison.
"""

import numpy as np
import pandas as pd
from src.data_preprocessor import DataPreprocessor
from src.models import TuitionFeePredictor
from src.evaluation import ModelEvaluator

def test_fairness():
    """Test that all models are compared fairly."""
    print("Testing fairness in model comparison...")
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(data_path="./Data/data.csv")
    X, y = preprocessor.preprocess()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print(f"Data shapes: X_train={X_train_scaled.shape}, y_train={y_train.shape}, X_test={X_test_scaled.shape}, y_test={y_test.shape}")
    
    # Initialize predictor
    predictor = TuitionFeePredictor()
    
    # Train all models
    predictor.fit_models(X_train_scaled, y_train)
    
    # Make predictions with all models
    all_predictions = predictor.predict(X_test_scaled)
    
    # Verify all models produced predictions
    print("Models and prediction shapes:")
    for name, pred in all_predictions.items():
        print(f"  {name}: {pred.shape}")
        
    # Verify all predictions have the same shape
    shapes = [pred.shape for pred in all_predictions.values()]
    if len(set(shapes)) == 1:
        print("[PASS] All models produced predictions with consistent shapes")
    else:
        print("[FAIL] Models produced predictions with inconsistent shapes:", shapes)
        
    # Evaluate all models fairly
    evaluator = ModelEvaluator()
    results_for_evaluation = {}
    
    for name in predictor.trained_models.keys():
        results_for_evaluation[name] = {
            'predictions': all_predictions[name],
            'y_true': y_test.values if hasattr(y_test, 'values') else y_test
        }
    
    # Compare models
    comparison_df = evaluator.compare_models(results_for_evaluation)
    
    print("\nModel Comparison Results:")
    print(comparison_df[['Model', 'MSE', 'RMSE', 'MAE', 'R2']])
    
    # Check if all models were evaluated
    expected_models = set(predictor.trained_models.keys())
    evaluated_models = set(comparison_df['Model'])
    
    if expected_models == evaluated_models:
        print("[PASS] All models were evaluated fairly")
    else:
        print("[FAIL] Model evaluation inconsistency:")
        print(f"  Expected: {expected_models}")
        print(f"  Evaluated: {evaluated_models}")
        
    return comparison_df

if __name__ == "__main__":
    comparison_df = test_fairness()