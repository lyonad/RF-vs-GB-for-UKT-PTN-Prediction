#!/usr/bin/env python3
"""
Script to verify the project is working correctly.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder

def main():
    print("Verifying project components...")
    
    # Check data loading
    try:
        data = pd.read_csv('./Data/data.csv')
        print(f"Data loaded with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Identify target columns
    ukt_columns = [col for col in data.columns if col.startswith('UKT-')]
    print(f"Found UKT columns: {ukt_columns}")

    # Prepare features and targets
    feature_columns = [col for col in data.columns if col not in ukt_columns]
    X = data[feature_columns].copy()
    y = data[ukt_columns].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Handle missing values
    X = X.fillna(X.mean(numeric_only=True))
    y = y.fillna(y.mean(numeric_only=True))

    # Test model training
    try:
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10, random_state=42))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        print(f"Model training and prediction successful!")
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Error in model training/prediction: {e}")
        return

    print("\nAll components working correctly!")
    print("The project is ready for use.")

if __name__ == "__main__":
    main()