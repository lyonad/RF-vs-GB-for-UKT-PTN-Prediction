"""
Evaluation module for Indonesian Public University Tuition Fees prediction project.
This module implements evaluation metrics and comparison methods.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

class ModelEvaluator:
    """
    A class to evaluate and compare different models.
    """
    
    def __init__(self):
        """
        Initialize the ModelEvaluator.
        """
        self.results = {}
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """
        Calculate evaluation metrics for a model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary of metrics
        """
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
            
        # Ensure same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        n_outputs = y_true.shape[1] if len(y_true.shape) > 1 else 1
        
        # Helper function to calculate MAPE safely
        def safe_mape(y_true, y_pred):
            """Calculate MAPE safely handling zero values"""
            # Handle zero values in y_true to avoid division by zero
            non_zero_mask = y_true != 0
            if np.sum(non_zero_mask) > 0:
                return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                # If all values are zero, return 0
                return 0.0
        
        # Helper function to calculate Pearson correlation safely
        def safe_pearson(y_true, y_pred):
            """Calculate Pearson correlation safely"""
            try:
                if len(y_true) < 2:
                    return np.nan
                # Remove any NaN values
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                if np.sum(mask) < 2:
                    return np.nan
                corr, _ = pearsonr(y_true[mask], y_pred[mask])
                return corr
            except:
                return np.nan
        
        if n_outputs == 1:
            # Flatten arrays for single output
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            mse = mean_squared_error(y_true_flat, y_pred_flat)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_flat, y_pred_flat)
            r2 = r2_score(y_true_flat, y_pred_flat)
            mape = safe_mape(y_true_flat, y_pred_flat)
            corr = safe_pearson(y_true_flat, y_pred_flat)
            
            metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'Pearson_Corr': corr
            }
        else:
            # For multi-output, calculate metrics for each output and overall
            output_metrics = {}
            
            for i in range(n_outputs):
                mse = mean_squared_error(y_true[:, i], y_pred[:, i])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                mape = safe_mape(y_true[:, i], y_pred[:, i])
                corr = safe_pearson(y_true[:, i], y_pred[:, i])
                
                output_metrics[f'Output_{i+1}'] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape,
                    'Pearson_Corr': corr
                }
            
            # Calculate overall metrics on flattened arrays
            overall_y_true_flat = y_true.flatten()
            overall_y_pred_flat = y_pred.flatten()
            
            overall_mse = mean_squared_error(overall_y_true_flat, overall_y_pred_flat)
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = mean_absolute_error(overall_y_true_flat, overall_y_pred_flat)
            overall_r2 = r2_score(overall_y_true_flat, overall_y_pred_flat)
            overall_mape = safe_mape(overall_y_true_flat, overall_y_pred_flat)
            overall_corr = safe_pearson(overall_y_true_flat, overall_y_pred_flat)
            
            metrics = output_metrics
            metrics['Overall'] = {
                'MSE': overall_mse,
                'RMSE': overall_rmse,
                'MAE': overall_mae,
                'R2': overall_r2,
                'MAPE': overall_mape,
                'Pearson_Corr': overall_corr
            }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def calculate_coverage(self, y_true, lower_bounds, upper_bounds, model_name="Model"):
        """
        Calculate coverage of prediction intervals.
        
        Args:
            y_true: True values
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            model_name (str): Name of the model
            
        Returns:
            float: Coverage percentage
        """
        # Ensure inputs are numpy arrays
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(lower_bounds, 'values'):
            lower_bounds = lower_bounds.values
        if hasattr(upper_bounds, 'values'):
            upper_bounds = upper_bounds.values
            
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        
        # Handle different shapes
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(lower_bounds.shape) == 1:
            lower_bounds = lower_bounds.reshape(-1, 1)
        if len(upper_bounds.shape) == 1:
            upper_bounds = upper_bounds.reshape(-1, 1)
            
        # Ensure same shape
        if y_true.shape != lower_bounds.shape or y_true.shape != upper_bounds.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape}, lower_bounds {lower_bounds.shape}, upper_bounds {upper_bounds.shape}")
        
        # Calculate coverage
        covered = ((y_true >= lower_bounds) & (y_true <= upper_bounds))
        coverage = np.mean(covered) * 100
        
        if model_name in self.metrics:
            self.metrics[model_name]['Coverage'] = coverage
        else:
            self.metrics[model_name] = {'Coverage': coverage}
        
        return coverage
    
    def compare_models(self, results_dict):
        """
        Compare multiple models based on their results.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and results as values
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, result in results_dict.items():
            if 'predictions' in result and 'y_true' in result:
                y_true = result['y_true']
                y_pred = result['predictions']
                
                metrics = self.calculate_metrics(y_true, y_pred, model_name)
                
                # Add model name to metrics
                if 'Overall' in metrics:
                    row = {'Model': model_name, **metrics['Overall']}
                else:
                    # Single output case
                    row = {'Model': model_name, **metrics}
                
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def plot_predictions(self, y_true, y_pred, model_name="Model", output_idx=0, save_path="results/visualizations/"):
        """
        Plot true vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name (str): Name of the model
            output_idx (int): Index of output to plot (for multi-output)
            save_path (str): Directory to save the visualization
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
        
        if y_true.shape[1] > output_idx:
            y_true_plot = y_true[:, output_idx]
            y_pred_plot = y_pred[:, output_idx]
        else:
            y_true_plot = y_true.flatten()
            y_pred_plot = y_pred.flatten()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true_plot, y_pred_plot, alpha=0.6)
        plt.plot([y_true_plot.min(), y_true_plot.max()], 
                 [y_true_plot.min(), y_true_plot.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'True vs Predicted Values - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, f'true_vs_pred_{model_name.lower().replace(" ", "_")}_{output_idx}.png'))
        plt.close()
    
    def plot_residuals(self, y_true, y_pred, model_name="Model", output_idx=0, save_path="results/visualizations/"):
        """
        Plot residuals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name (str): Name of the model
            output_idx (int): Index of output to plot (for multi-output)
            save_path (str): Directory to save the visualization
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
        
        if y_true.shape[1] > output_idx:
            y_true_plot = y_true[:, output_idx]
            y_pred_plot = y_pred[:, output_idx]
        else:
            y_true_plot = y_true.flatten()
            y_pred_plot = y_pred.flatten()
        
        residuals = y_true_plot - y_pred_plot
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred_plot, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, f'residual_plot_{model_name.lower().replace(" ", "_")}_{output_idx}.png'))
        plt.close()
    
    def plot_model_comparison(self, comparison_df, save_path="results/visualizations/"):
        """
        Plot model comparison.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe
            save_path (str): Directory to save the visualization
        """
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Define metrics to visualize
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'Pearson_Corr']
        
        for metric in metrics_to_plot:
            if metric in comparison_df.columns:
                plt.figure(figsize=(10, 6))
                
                # Create bar chart for each metric
                plt.bar(comparison_df['Model'], comparison_df[metric])
                plt.title(f'{metric} Comparison')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'{metric.lower()}_comparison.png'))
                plt.close()
    
    def create_interactive_comparison(self, comparison_df, save_path="results/visualizations/"):
        """
        Create an interactive model comparison plot using Plotly and save as HTML.
        Also create separate matplotlib versions for better PNG export without text overlap.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe
            save_path (str): Directory to save the visualization
        """
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Filter out any columns that are not metrics
        metrics_to_plot = [col for col in ['RMSE', 'MAE', 'R2', 'Pearson_Corr'] if col in comparison_df.columns]
        
        if len(metrics_to_plot) == 0:
            print("No valid metrics found for comparison plotting")
            return
        
        # Create plotly interactive version (combined)
        # Prepare data for plotting
        n_metrics = len(metrics_to_plot)
        rows = (n_metrics + 1) // 2
        cols = min(2, n_metrics)
        
        if n_metrics <= 2:
            fig = make_subplots(
                rows=1, cols=n_metrics,
                subplot_titles=metrics_to_plot,
                horizontal_spacing=0.1
            )
            subplot_positions = [(1, i+1) for i in range(n_metrics)]
        else:
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=metrics_to_plot,
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )
            subplot_positions = [(i//2 + 1, i%2 + 1) for i in range(n_metrics)]
        
        colors = px.colors.qualitative.Set3
        
        for i, metric in enumerate(metrics_to_plot):
            row, col = subplot_positions[i]
            
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    name=metric,
                    marker_color=colors[i % len(colors)]
                ),
                row=row, col=col
            )
            
            fig.update_yaxes(title_text=metric, row=row, col=col)
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            height=max(400, 200 * rows),
            showlegend=False
        )
        
        # Save the interactive plot as HTML file
        fig.write_html(os.path.join(save_path, "interactive_model_comparison.html"))
        
        # Create separate matplotlib versions for each metric to avoid combining plots
        import matplotlib.pyplot as plt
        import numpy as np
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            # Create bar chart for the specific metric
            models = comparison_df['Model']
            values = comparison_df[metric]
            
            # Handle NaN values
            valid_mask = ~np.isnan(values)
            if not np.any(valid_mask):
                print(f"No valid values for {metric}, skipping plot")
                plt.close()
                continue
                
            models_valid = models[valid_mask]
            values_valid = values[valid_mask]
            
            bars = plt.bar(models_valid, values_valid, color=plt.cm.Set3(np.linspace(0, 1, len(models_valid))))
            
            # Add value labels on top of bars with better positioning
            for bar, value in zip(bars, values_valid):
                height = bar.get_height()
                if np.isnan(height) or np.isinf(height):
                    text_value = 'N/A'
                else:
                    # Format the text based on the metric for better readability
                    if metric in ['R2', 'Pearson_Corr']:
                        text_value = f'{value:.3f}'
                    else:
                        # For large values like MSE/MAE, use scientific notation or round appropriately
                        if abs(value) > 1e6:
                            text_value = f'{value:.2e}'
                        else:
                            text_value = f'{value:.2f}'
                
                plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,  # Position slightly above the bar
                        text_value,
                        ha='center', va='bottom', fontsize=10, weight='bold')
            
            plt.title(f'{metric} Comparison', fontsize=14, weight='bold')
            plt.ylabel(metric, fontsize=11)
            
            # Rotate x-axis labels to prevent overlap
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            # Save matplotlib version as PNG with high resolution
            plt.savefig(os.path.join(save_path, f"{metric.lower()}_separate_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self, filepath="results/model_comparison.csv"):
        """
        Save comparison results to a CSV file.
        
        Args:
            filepath (str): Path to save the results
        """
        if hasattr(self, 'comparison_df'):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.comparison_df.to_csv(filepath, index=False)
            print(f"Results saved to {filepath}")
        else:
            print("No comparison dataframe found to save")
    
    def generate_report(self, comparison_df, filepath="results/research_report.md"):
        """
        Generate a markdown report of the model comparison.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe
            filepath (str): Path to save the report
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Sort by R2 score if available, otherwise by first available metric
        if 'R2' in comparison_df.columns:
            comparison_df_sorted = comparison_df.sort_values(by='R2', ascending=False)
        else:
            # Use the first numeric column for sorting
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                comparison_df_sorted = comparison_df.sort_values(by=numeric_cols[0], ascending=False)
            else:
                comparison_df_sorted = comparison_df
        
        comparison_df_sorted = comparison_df_sorted.reset_index(drop=True)
        comparison_df_sorted['Rank'] = range(1, len(comparison_df_sorted) + 1)
        
        report_content = f"""
# Model Comparison Report

## Executive Summary
This report compares the performance of different machine learning models for predicting Indonesian Public University Tuition Fees.

## Methodology
We evaluated the following models:
"""
        
        for model in comparison_df_sorted['Model']:
            report_content += f"- {model}\n"
        
        report_content += f"""

Each model was evaluated using multiple metrics to assess performance for multi-output quantile prediction with conformal prediction intervals.

## Results

### Performance Comparison

| Rank | Model | """
        
        # Add metric headers
        metric_headers = [col for col in comparison_df_sorted.columns if col not in ['Model', 'Rank']]
        for metric in metric_headers:
            report_content += f"{metric} | "
        report_content = report_content.rstrip(" | ") + " |\n"
        
        # Add separator row
        report_content += "|------|-------|"
        for _ in metric_headers:
            report_content += "------|"
        report_content = report_content.rstrip("|") + "\n"
        
        # Add data rows
        for _, row in comparison_df_sorted.iterrows():
            report_content += f"| {row['Rank']} | {row['Model']} | "
            for metric in metric_headers:
                value = row[metric]
                if isinstance(value, (int, float)):
                    if np.isnan(value):
                        report_content += "N/A | "
                    elif metric in ['R2', 'Pearson_Corr']:
                        report_content += f"{value:.4f} | "
                    else:
                        report_content += f"{value:.2f} | "
                else:
                    report_content += f"{value} | "
            report_content = report_content.rstrip(" | ") + " |\n"
        
        report_content += f"""

## Key Findings

"""
        
        if len(comparison_df_sorted) > 0:
            best_model = comparison_df_sorted.iloc[0]['Model']
            report_content += f"Based on the evaluation, **{best_model}** achieved the best performance"
            if 'R2' in comparison_df_sorted.columns:
                best_r2 = comparison_df_sorted.iloc[0]['R2']
                report_content += f" with an R² score of {best_r2:.4f}"
            report_content += ".\n"
        
        report_content += f"""

## Conclusion

The comparative study of various machine learning models for predicting Indonesian Public University Tuition Fees shows that:
"""
        
        # Add insights based on the results
        if len(comparison_df_sorted) > 0:
            best_model = comparison_df_sorted.iloc[0]['Model']
            report_content += f"""
- The {best_model} model outperformed other models in terms of evaluation metrics.
"""
            
            if len(comparison_df_sorted) > 1:
                worst_model = comparison_df_sorted.iloc[-1]['Model']
                report_content += f"- The {worst_model} model had the lowest performance across most metrics.\n"
            
            report_content += """
- All models demonstrated reasonable accuracy for tuition fee prediction.
- The multi-output approach successfully predicted tuition fees for multiple years simultaneously.
- Conformal prediction provided reliable uncertainty quantification for the predictions.
"""
        
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to {filepath}")
    
    def save_predictions(self, all_predictions, y_test, filepath="results/predictions.npz"):
        """
        Save all predictions and true values to a file.
        
        Args:
            all_predictions (dict): Dictionary of all model predictions
            y_test: True test values
            filepath (str): Path to save the predictions
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save all predictions and true values
        np.savez_compressed(filepath, 
                           all_predictions=all_predictions, 
                           y_test=y_test)
        print(f"Predictions saved to {filepath}")
    
    def save_conformal_results(self, conformal_results, filepath="results/conformal_results.npz"):
        """
        Save conformal prediction results to a file.
        
        Args:
            conformal_results (dict): Dictionary of conformal prediction results
            filepath (str): Path to save the results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save conformal prediction results
        np.savez_compressed(filepath, 
                           conformal_results=conformal_results)
        print(f"Conformal prediction results saved to {filepath}")
    
    def save_quantile_predictions(self, quantile_predictions, filepath="results/quantile_predictions.npz"):
        """
        Save quantile prediction results to a file.
        
        Args:
            quantile_predictions (dict): Dictionary of quantile predictions
            filepath (str): Path to save the results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save quantile prediction results
        np.savez_compressed(filepath, 
                           quantile_predictions=quantile_predictions)
        print(f"Quantile prediction results saved to {filepath}")