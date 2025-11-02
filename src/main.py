"""
Main execution script for the Comparative Study of Random Forest and Gradient-Boosted Trees 
(XGBoost, CatBoost, LightGBM) for Predicting Indonesian Public University Tuition Fees 
with Multi-Output Quantile and Conformal Prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src and utils directories to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_preprocessor import DataPreprocessor
from models import TuitionFeePredictor
from conformal_prediction import create_conformal_predictors, MultiOutputConformalPredictor
from evaluation import ModelEvaluator
from utils.visualization import (
    visualize_data_distribution, 
    visualize_feature_importance, 
    plot_prediction_intervals,
    plot_quantile_predictions,
    correlation_heatmap,
    analyze_prediction_accuracy
)

def main():
    """
    Main function to execute the research project.
    """
    print("Starting Indonesian Public University Tuition Fees Prediction Research...")
    print("=" * 70)
    
    # Step 1: Data Preprocessing
    print("\nStep 1: Data Preprocessing")
    print("-" * 30)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(data_path="./Data/data.csv")
    data = preprocessor.load_data()
    
    # Display basic info about the dataset
    print(f"Dataset shape: {data.shape}")
    print(f"Dataset columns: {list(data.columns)}")
    print("\nFirst few rows:")
    print(data.head())
    
    # Visualize data distribution
    # Identify UKT columns from the actual data
    ukt_columns = [col for col in data.columns if col.startswith('UKT-')]
    if len(ukt_columns) == 0:
        # If no UKT columns found, use a default list
        ukt_columns = ['UKT-1', 'UKT-2', 'UKT-3', 'UKT-4', 'UKT-5']  # Using first 5 UKT columns
    
    visualize_data_distribution(data, ukt_columns, save_path="results/visualizations/")
    
    # Show correlation heatmap
    correlation_heatmap(data, ukt_columns, save_path="results/visualizations/")
    
    # Preprocess the data
    X, y = preprocessor.preprocess(target_columns=ukt_columns)
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Step 2: Model Training
    print("\nStep 2: Model Training")
    print("-" * 30)
    
    # Initialize the predictor
    predictor = TuitionFeePredictor()
    
    # Train all models
    predictor.fit_models(X_train_scaled, y_train)
    
    # Step 3: Model Prediction
    print("\nStep 3: Model Prediction")
    print("-" * 30)
    
    # Make predictions with all models
    all_predictions = predictor.predict(X_test_scaled)
    print(f"Made predictions with {len(all_predictions)} models")
    
    # Make quantile predictions
    quantile_predictions = predictor.predict_quantiles(X_test_scaled)
    print(f"Made quantile predictions with {len(quantile_predictions)} models")
    
    # Step 4: Conformal Prediction
    print("\nStep 4: Conformal Prediction")
    print("-" * 30)
    
    # Create conformal predictors
    conformal_predictors = create_conformal_predictors(predictor.trained_models)
    
    # Fit conformal predictors and make predictions
    conformal_results = {}
    for name, cp in conformal_predictors.items():
        print(f"Fitting conformal predictor for {name}...")
        cp.fit(X_train_scaled, y_train)
        
        # Make conformal predictions
        y_pred_conf, lower_bounds, upper_bounds = cp.predict(X_test_scaled)
        conformal_results[name] = {
            'predictions': y_pred_conf,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds
        }
    
    # Step 5: Model Evaluation
    print("\nStep 5: Model Evaluation")
    print("-" * 30)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Prepare results for evaluation
    results_for_evaluation = {}
    for name in predictor.trained_models.keys():
        results_for_evaluation[name] = {
            'predictions': all_predictions[name],
            'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
            'lower_bounds': conformal_results[name]['lower_bounds'] if name in conformal_results else None,
            'upper_bounds': conformal_results[name]['upper_bounds'] if name in conformal_results else None
        }
    
    # Compare models
    comparison_df = evaluator.compare_models(results_for_evaluation)
    print("\nModel Comparison Results:")
    print(comparison_df)
    
    # Calculate coverage for conformal prediction
    for name in predictor.trained_models.keys():
        if name in conformal_results:
            coverage = evaluator.calculate_coverage(
                y_test.values if hasattr(y_test, 'values') else y_test,
                conformal_results[name]['lower_bounds'],
                conformal_results[name]['upper_bounds'],
                name
            )
            print(f"{name} - Prediction Interval Coverage: {coverage:.2f}%")
    
    # Step 6: Visualization and Analysis
    print("\nStep 6: Visualization and Analysis")
    print("-" * 40)
    
    # Plot model comparison
    evaluator.plot_model_comparison(comparison_df, save_path="results/visualizations/")
    
    # Create interactive comparison
    evaluator.create_interactive_comparison(comparison_df, save_path="results/visualizations/")
    
    # Visualize feature importance
    visualize_feature_importance(predictor.trained_models, X.columns.tolist(), save_path="results/visualizations/")
    
    # Plot prediction intervals for a sample model
    sample_model = list(conformal_results.keys())[0]
    plot_prediction_intervals(
        y_test.values if hasattr(y_test, 'values') else y_test,
        conformal_results[sample_model]['predictions'],
        conformal_results[sample_model]['lower_bounds'],
        conformal_results[sample_model]['upper_bounds'],
        sample_model,
        save_path="results/visualizations/"
    )
    
    # Plot quantile predictions for a sample model
    sample_quantile_model = list(quantile_predictions.keys())[0]
    plot_quantile_predictions(
        quantile_predictions[sample_quantile_model],
        y_test.values if hasattr(y_test, 'values') else y_test,
        sample_quantile_model,
        save_path="results/visualizations/"
    )
    
    # Analyze prediction accuracy for the best model
    best_model = comparison_df.loc[comparison_df['R2'].idxmax(), 'Model']
    best_predictions = all_predictions[best_model]
    
    accuracy_analysis = analyze_prediction_accuracy(
        y_test.values if hasattr(y_test, 'values') else y_test,
        best_predictions,
        save_path="results/"
    )
    
    print(f"\nAccuracy Analysis for Best Model ({best_model}):")
    for metric, value in accuracy_analysis.items():
        print(f"{metric}: {value:.4f}")
    
    # Step 7: Save Results
    print("\nStep 7: Saving Results")
    print("-" * 25)
    
    # Save comparison results
    evaluator.comparison_df = comparison_df  # Store for saving
    evaluator.save_results("results/model_comparison.csv")
    
    # Generate report
    evaluator.generate_report(comparison_df, "results/research_report.md")
    
    # Save all models
    for model_name in predictor.trained_models.keys():
        predictor.save_model(model_name, f"models/{model_name.lower().replace(' ', '_')}_model.pkl")
    
    # Note: We no longer save a separate 'best_*.pkl' file. The per-model file
    # (e.g., models/catboost_model.pkl) is the canonical saved model.
    
    # Save conformal predictors
    from conformal_prediction import save_conformal_predictors
    save_conformal_predictors(conformal_predictors)
    
    print("\n" + "=" * 70)
    print("Research project completed successfully!")
    print(f"Best performing model: {best_model}")
    print("Results saved to the 'results' directory")
    print("Trained models saved to the 'models' directory")
    print("=" * 70)

if __name__ == "__main__":
    main()