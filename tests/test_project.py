"""
Test suite for Indonesian Public University Tuition Fees prediction project.
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessor import DataPreprocessor
from models import TuitionFeePredictor, MultiOutputQuantileRegressor
from conformal_prediction import ConformalPredictor, MultiOutputConformalPredictor
from evaluation import ModelEvaluator


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Use a different data path for tests to avoid file conflicts
        self.preprocessor = DataPreprocessor(data_path="tests/test_data.csv")
        self.data = self.preprocessor._create_sample_data()
    
    def test_load_data(self):
        """Test data loading functionality."""
        # Test with sample data
        X, y = self.preprocessor.preprocess()
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.DataFrame)
        self.assertEqual(X.shape[0], y.shape[0])
    
    def test_preprocess(self):
        """Test data preprocessing functionality."""
        X, y = self.preprocessor.preprocess()
        self.assertGreaterEqual(X.shape[1], 1)  # At least one feature
        self.assertGreaterEqual(y.shape[1], 1)  # At least one target
    
    def test_split_data(self):
        """Test data splitting functionality."""
        X, y = self.preprocessor.preprocess()
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        self.assertEqual(X_train.shape[0] + X_test.shape[0], X.shape[0])
        self.assertEqual(y_train.shape[0] + y_test.shape[0], y.shape[0])
    
    def test_scale_features(self):
        """Test feature scaling functionality."""
        X, y = self.preprocessor.preprocess()
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # Check that shapes are preserved
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)


class TestTuitionFeePredictor(unittest.TestCase):
    """Test cases for TuitionFeePredictor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.predictor = TuitionFeePredictor()
        
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.rand(50, 8)
        self.y_train = np.random.rand(50, 4)  # 4 outputs
        self.X_test = np.random.rand(20, 8)
        self.y_test = np.random.rand(20, 4)  # 4 outputs
    
    def test_fit_models(self):
        """Test model training functionality."""
        self.predictor.fit_models(self.X_train, self.y_train)
        
        # Check that models are trained
        self.assertEqual(len(self.predictor.trained_models), len(self.predictor.models))
        self.assertEqual(len(self.predictor.quantile_models), len(self.predictor.models))
    
    def test_predict(self):
        """Test prediction functionality."""
        self.predictor.fit_models(self.X_train, self.y_train)
        predictions = self.predictor.predict(self.X_test)
        
        # Check that predictions have correct format
        for name, pred in predictions.items():
            self.assertEqual(pred.shape, self.y_test.shape)
    
    def test_predict_quantiles(self):
        """Test quantile prediction functionality."""
        self.predictor.fit_models(self.X_train, self.y_train)
        quantile_predictions = self.predictor.predict_quantiles(self.X_test)
        
        # Check that quantile predictions have correct format
        for name, quantile_pred in quantile_predictions.items():
            for q, pred in quantile_pred.items():
                self.assertEqual(pred.shape, self.y_test.shape)


class TestConformalPredictor(unittest.TestCase):
    """Test cases for ConformalPrediction class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        
        self.base_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10, random_state=42))
        
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.rand(50, 8)
        self.y_train = np.random.rand(50, 2)  # 2 outputs for simplicity
        self.X_test = np.random.rand(20, 8)
        self.y_test = np.random.rand(20, 2)
    
    def test_conformal_predictor(self):
        """Test basic conformal prediction functionality."""
        # Test single-output conformal predictor
        from sklearn.ensemble import RandomForestRegressor
        single_model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        cp = ConformalPredictor(single_model, significance_level=0.1)
        
        # Fit with single output
        y_train_1d = self.y_train[:, 0]  # Take first output
        cp.fit(self.X_train, y_train_1d)
        
        y_pred, lower, upper = cp.predict(self.X_test)
        
        # Check shapes
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],))
        self.assertEqual(lower.shape, (self.X_test.shape[0],))
        self.assertEqual(upper.shape, (self.X_test.shape[0],))
        
        # Check that intervals are valid
        self.assertTrue(np.all(lower <= upper))
    
    def test_multi_output_conformal_predictor(self):
        """Test multi-output conformal prediction functionality."""
        mocp = MultiOutputConformalPredictor(self.base_model, significance_level=0.1)
        mocp.fit(self.X_train, self.y_train)
        
        y_pred, lower, upper = mocp.predict(self.X_test)
        
        # Check shapes
        self.assertEqual(y_pred.shape, self.y_test.shape)
        self.assertEqual(lower.shape, self.y_test.shape)
        self.assertEqual(upper.shape, self.y_test.shape)
        
        # Check that intervals are valid
        self.assertTrue(np.all(lower <= upper))


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.evaluator = ModelEvaluator()
        
        # Create sample data
        np.random.seed(42)
        self.y_true = np.random.rand(50, 3)  # 3 outputs
        self.y_pred = np.random.rand(50, 3)
    
    def test_calculate_metrics(self):
        """Test metric calculation functionality."""
        metrics = self.evaluator.calculate_metrics(self.y_true, self.y_pred, "TestModel")
        
        # Check that required metrics are present
        self.assertIn('MSE', metrics['Overall'])
        self.assertIn('RMSE', metrics['Overall'])
        self.assertIn('MAE', metrics['Overall'])
        self.assertIn('R2', metrics['Overall'])
        self.assertIn('Pearson_Corr', metrics['Overall'])
    
    def test_calculate_coverage(self):
        """Test coverage calculation functionality."""
        # Create some sample intervals
        lower_bounds = self.y_true - 1
        upper_bounds = self.y_true + 1
        
        coverage = self.evaluator.calculate_coverage(
            self.y_true, lower_bounds, upper_bounds, "TestModel"
        )
        
        # Since intervals are very wide, coverage should be 100%
        self.assertEqual(coverage, 100.0)
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create sample results dictionary
        results_dict = {
            'Model1': {
                'predictions': self.y_pred,
                'y_true': self.y_true
            },
            'Model2': {
                'predictions': self.y_pred * 1.1,  # Slightly different predictions
                'y_true': self.y_true
            }
        }
        
        comparison_df = self.evaluator.compare_models(results_dict)
        
        # Check that comparison dataframe has correct format
        self.assertEqual(len(comparison_df), 2)
        self.assertIn('Model', comparison_df.columns)
        self.assertIn('R2', comparison_df.columns)


if __name__ == '__main__':
    print("Running tests for Indonesian Public University Tuition Fees Prediction Project...")
    unittest.main(verbosity=2)