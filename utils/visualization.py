"""
Utility module for Indonesian Public University Tuition Fees prediction project.
This module contains helper functions for visualization, data analysis, and other utilities.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
from typing import List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def visualize_data_distribution(data: pd.DataFrame, target_cols: List[str], save_path: str = "results/visualizations/"):
    """
    Visualize the distribution of target variables.
    
    Args:
        data (pd.DataFrame): The dataset
        target_cols (List[str]): List of target column names
        save_path (str): Directory to save the visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Select only numeric target columns
    numeric_target_cols = [col for col in target_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
    
    if not numeric_target_cols:
        print("No numeric target columns found for visualization.")
        return
    
    # Create separate visualization for each numeric target column
    for col in numeric_target_cols:
        # Remove NaN values for plotting
        col_data = data[col].dropna()
        
        plt.figure(figsize=(8, 6))
        plt.hist(col_data, bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'data_distribution_{col}.png'))
        plt.close()

def visualize_feature_importance(models: dict, feature_names: List[str], top_n: int = 10, save_path: str = "results/visualizations/"):
    """
    Visualize feature importance for different models.
    
    Args:
        models (dict): Dictionary of trained models
        feature_names (List[str]): List of feature names
        top_n (int): Number of top features to show
        save_path (str): Directory to save the visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    for name, model in models.items():
        # Get feature importances
        if hasattr(model, 'estimators_'):
            # For ensemble models with multiple outputs
            importances = []
            for estimator in model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            # Average importances across outputs
            if len(importances) > 0:
                avg_importances = np.mean(importances, axis=0)
            else:
                print(f"No feature importances found for {name}")
                continue
        elif hasattr(model, 'feature_importances_'):
            avg_importances = model.feature_importances_
        else:
            print(f"Feature importances not available for {name}")
            continue
        
        # Get top features
        indices = np.argsort(avg_importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices if i < len(feature_names)]
        top_importances = avg_importances[indices[:len(top_features)]]
        
        # Create separate plot for each model
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {name}')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'feature_importance_{name.lower().replace(" ", "_")}.png'))
        plt.close()

def plot_prediction_intervals(y_true, y_pred, lower_bounds, upper_bounds, model_name="Model", n_samples=100, save_path: str = "results/visualizations/"):
    """
    Plot prediction intervals for a model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        model_name (str): Name of the model
        n_samples (int): Number of samples to plot
        save_path (str): Directory to save the visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # For visualization, we'll focus on the first output if multi-output
    if len(y_true.shape) > 1:
        y_true = y_true[:n_samples, 0]
        y_pred = y_pred[:n_samples, 0]
        lower_bounds = lower_bounds[:n_samples, 0]
        upper_bounds = upper_bounds[:n_samples, 0]
    else:
        y_true = y_true[:n_samples]
        y_pred = y_pred[:n_samples]
        lower_bounds = lower_bounds[:n_samples]
        upper_bounds = upper_bounds[:n_samples]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot true values
    ax.plot(range(len(y_true)), y_true, label='True Values', color='blue', marker='o', linestyle='', markersize=4)
    
    # Plot predicted values with intervals
    ax.plot(range(len(y_pred)), y_pred, label='Predicted Values', color='red', marker='x', linestyle='', markersize=5)
    ax.fill_between(range(len(y_pred)), lower_bounds, upper_bounds, color='gray', alpha=0.2, label='Prediction Interval')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value')
    ax.set_title(f'Prediction Intervals - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'prediction_intervals_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()

def plot_quantile_predictions(quantile_predictions: dict, y_true, model_name: str, save_path: str = "results/visualizations/"):
    """
    Plot quantile predictions for a model.
    
    Args:
        quantile_predictions (dict): Dictionary of quantile predictions
        y_true: True values
        model_name (str): Name of the model
        save_path (str): Directory to save the visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Handle different quantiles
    quantiles = [0.1, 0.5, 0.9]
    colors = ['red', 'orange', 'green']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For visualization, we'll focus on the first output if multi-output
    if len(y_true.shape) > 1:
        y_true_plot = y_true[:, 0]
    else:
        y_true_plot = y_true
    
    # Plot true values
    ax.scatter(range(len(y_true_plot)), y_true_plot, label='True Values', color='blue', alpha=0.6)
    
    # Plot different quantiles
    for i, q in enumerate(quantiles):
        if q in quantile_predictions:
            pred_q = quantile_predictions[q]
            if len(pred_q.shape) > 1:
                pred_q = pred_q[:, 0]
            
            ax.plot(range(len(pred_q)), pred_q, label=f'Quantile {q}', color=colors[i], linewidth=2)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value')
    ax.set_title(f'Quantile Predictions - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'quantile_predictions_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()

def correlation_heatmap(data: pd.DataFrame, target_cols: List[str], save_path: str = "results/visualizations/"):
    """
    Create a correlation heatmap for the dataset.
    
    Args:
        data (pd.DataFrame): The dataset
        target_cols (List[str]): List of target column names
        save_path (str): Directory to save the visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=[np.number])
    
    # If target columns are specified and exist in the data, calculate correlation for them
    if target_cols:
        available_targets = [col for col in target_cols if col in numeric_data.columns]
        if available_targets:
            # Calculate correlation matrix for numeric columns including targets
            corr_matrix = numeric_data[available_targets].corr()
        else:
            # If target columns are not in numeric data, use all numeric columns
            corr_matrix = numeric_data.corr()
    else:
        # If no target columns specified, use all numeric columns
        corr_matrix = numeric_data.corr()
    
    # Create the heatmap
    if not corr_matrix.empty:
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle for cleaner look
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'correlation_heatmap.png'))
        plt.close()
    else:
        print("No numeric columns available for correlation analysis")

def save_model_results(results: dict, filepath: str = "results/model_results.npy"):
    """
    Save model results to a file.
    
    Args:
        results (dict): Dictionary of model results
        filepath (str): Path to save the results
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, results)
    print(f"Model results saved to {filepath}")

def load_model_results(filepath: str = "results/model_results.npy"):
    """
    Load model results from a file.
    
    Args:
        filepath (str): Path to load the results from
        
    Returns:
        dict: Dictionary of model results
    """
    if os.path.exists(filepath):
        results = np.load(filepath, allow_pickle=True).item()
        print(f"Model results loaded from {filepath}")
        return results
    else:
        print(f"File {filepath} does not exist")
        return {}

def format_currency(value: float) -> str:
    """
    Format a value as Indonesian Rupiah.
    
    Args:
        value (float): The value to format
        
    Returns:
        str: Formatted currency string
    """
    return f"Rp {value:,.0f}".replace(",", ".")

def analyze_prediction_accuracy(y_true, y_pred, threshold_percentage=0.1, save_path: str = "results/"):
    """
    Analyze the accuracy of predictions within a certain threshold.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        threshold_percentage (float): Threshold as a percentage of true value
        save_path (str): Directory to save the analysis results
        
    Returns:
        dict: Analysis results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Ensure inputs are numpy arrays
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle different shapes
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Handle zero values in y_true to avoid division by zero in MAPE calculation
    # For MAPE calculation, we'll exclude zero values or use a meaningful reference value
    non_zero_mask = y_true != 0
    
    # Calculate errors
    errors = y_true - y_pred
    
    # Calculate absolute percentage errors only for non-zero true values
    if np.sum(non_zero_mask) > 0:
        abs_percentage_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100
        mape = np.mean(abs_percentage_errors)
        mdape = np.median(abs_percentage_errors)
    else:
        # If all values are zero, set MAPE to a default value
        mape = np.nan
        mdape = np.nan
    
    # Calculate percentage of predictions within threshold (using all data points)
    # For zero true values, we check if prediction is also close to zero (within absolute threshold)
    abs_errors = np.abs(errors)
    # For non-zero true values, use percentage threshold
    within_threshold_mask = np.zeros_like(y_true, dtype=bool)
    
    # For non-zero true values, use percentage threshold
    if np.sum(non_zero_mask) > 0:
        percentage_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        within_threshold_mask[non_zero_mask] = percentage_errors <= threshold_percentage
    
    # For zero true values, use absolute threshold (e.g., 10000 IDR as tolerance)
    zero_tolerance = 10000  # IDR tolerance for zero values
    zero_mask = ~non_zero_mask
    if np.sum(zero_mask) > 0:
        within_threshold_mask[zero_mask] = abs_errors[zero_mask] <= zero_tolerance
    
    within_threshold = np.mean(within_threshold_mask) * 100
    
    results = {
        'Within_Threshold_Percentage': within_threshold,
        'MAPE': mape,
        'MDAPE': mdape,
        'Mean_Error': np.mean(errors),
        'Std_Error': np.std(errors)
    }
    
    # Save the results to a file
    accuracy_results_path = os.path.join(save_path, 'accuracy_analysis.npy')
    np.save(accuracy_results_path, results)
    print(f"Accuracy analysis results saved to {accuracy_results_path}")
    
    return results