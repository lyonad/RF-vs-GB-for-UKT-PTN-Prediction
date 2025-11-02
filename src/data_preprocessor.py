"""
Data preprocessing module for Indonesian Public University Tuition Fees prediction project.
This module handles data loading, cleaning, feature engineering, and preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    """
    A class to preprocess data for tuition fee prediction.
    """
    
    def __init__(self, data_path: str = "./Data/data.csv"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.data = None
        
    def load_data(self):
        """
        Load the data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully with shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            # Create a sample dataset for demonstration
            self.data = self._create_sample_data()
            return self.data
    
    def _create_sample_data(self):
        """
        Create a sample dataset for demonstration purposes based on actual data structure.
        
        Returns:
            pd.DataFrame: Sample data
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Create features based on the actual data structure
        data = {
            'university_type': np.random.choice(['State', 'Private'], n_samples, p=[0.6, 0.4]),
            'location': np.random.choice(['Java', 'Sumatra', 'Kalimantan', 'Sulawesi', 'Others'], n_samples),
            'accreditation': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.5, 0.2]),
            'program_type': np.random.choice(['S1', 'S2', 'S3', 'D3', 'D4'], n_samples, p=[0.5, 0.15, 0.05, 0.15, 0.15]),
            'faculty': np.random.choice(['Engineering', 'Economics', 'Law', 'Medicine', 'Arts', 'Science', 'Computer Science'], n_samples),
            'urban_rural': np.random.choice(['Urban', 'Rural'], n_samples, p=[0.65, 0.35]),
            'region_development': np.random.uniform(0, 10, n_samples),  # Index of regional development
            'student_capacity': np.random.randint(100, 5000, n_samples),
            'years_since_established': np.random.randint(5, 60, n_samples),
            'admission_method': np.random.choice(['SNBP/SNBT', 'Mandiri'], n_samples, p=[0.7, 0.3])
        }
        
        # Create UKT columns (UKT-1 to UKT-11) similar to actual data structure
        base_fee = np.random.uniform(0, 15000000, n_samples)  # Base tuition fee in IDR
        
        # Apply factors based on features
        base_fee = base_fee * (1 + data['region_development'] / 50)  # Higher in developed regions
        base_fee = base_fee * (1 + (data['accreditation'] == 'A') * 0.5 + (data['accreditation'] == 'B') * 0.2)  # Accreditation affects fees
        base_fee = base_fee * (1 + (data['university_type'] == 'Private') * 1.0)  # Private universities more expensive
        base_fee = base_fee * (1 + (np.array(data['program_type']) == 'S2') * 0.3 + (np.array(data['program_type']) == 'S3') * 0.6)  # Advanced programs more expensive
        base_fee = base_fee * (1 + (np.array(data['faculty']) == 'Medicine') * 1.5)  # Medicine more expensive
        
        # Add random noise
        base_fee = base_fee * np.random.uniform(0.8, 1.2, n_samples)
        
        # Create UKT columns similar to actual data structure
        data['UKT-1'] = np.random.uniform(0, 5000000, n_samples)  # Often 0 for subsidized students
        data['UKT-2'] = base_fee * 0.6  # Lower than base fee for moderate financial capacity
        data['UKT-3'] = base_fee  # Main fee for most students
        data['UKT-4'] = base_fee * 1.05  # Slightly higher
        data['UKT-5'] = base_fee * 1.10  # Higher for better financial capacity
        data['UKT-6'] = base_fee * 1.15  # Higher for even better financial capacity
        data['UKT-7'] = base_fee * 1.30  # Higher category for higher financial capacity
        data['UKT-8'] = data['UKT-7'] * 1.05  # Even higher
        data['UKT-9'] = data['UKT-7'] * 1.10  # Continuing the progression
        data['UKT-10'] = data['UKT-7'] * 1.15  # Higher
        data['UKT-11'] = data['UKT-7'] * 1.20  # Highest category
        
        # Ensure that UKT values are non-decreasing as per typical structure for all UKT levels
        for i in range(2, 12):  # For UKT-2 to UKT-11
            col = f'UKT-{i}'
            col_prev = f'UKT-{i-1}'
            # Ensure current UKT is at least as high as previous one
            mask = data[col] < data[col_prev]
            data[col][mask] = data[col_prev][mask] * 1.05  # At least 5% higher
        
        df = pd.DataFrame(data)
        
        # Save sample data for future use
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path, index=False)
        print(f"Sample data created with shape: {df.shape}")
        return df
    
    def preprocess(self, target_columns=None):
        """
        Preprocess the data by handling missing values, encoding categorical variables, etc.
        
        Args:
            target_columns (list): List of target column names
            
        Returns:
            tuple: (X, y) where X is features and y is targets
        """
        if self.data is None:
            self.load_data()
        
        # If target_columns not provided, try to infer them
        if target_columns is None:
            # Look for columns that start with 'UKT-' (these are the tuition fee columns)
            target_columns = [col for col in self.data.columns if col.startswith('UKT-')]
            if not target_columns:
                # Default target columns if no UKT columns found
                target_columns = ['UKT-1', 'UKT-2', 'UKT-3', 'UKT-4', 'UKT-5']
        
        # Separate features and targets, ensuring target columns exist in the data
        available_targets = [col for col in target_columns if col in self.data.columns]
        if not available_targets:
            raise ValueError(f"Target columns {target_columns} not found in the dataset")
        
        feature_columns = [col for col in self.data.columns if col not in available_targets]
        X = self.data[feature_columns].copy()
        y = self.data[available_targets].copy()
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values if any
        X = X.fillna(X.mean(numeric_only=True))
        y = y.fillna(y.mean(numeric_only=True))
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split the data into train and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): Targets
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def scale_features(self, X_train, X_test):
        """
        Scale the features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Returns:
            tuple: (scaled_X_train, scaled_X_test)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled