"""
UKT Prediction Web Application
Professional Flask-based web interface for Indonesian Public University Tuition Fee Prediction
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import project modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.data_preprocessor import DataPreprocessor
from src.conformal_prediction import MultiOutputConformalPredictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ukt-prediction-secret-key-2025'

# Global variables for model and preprocessor
model = None
preprocessor = None
conformal_predictors = None
feature_names = None
label_encoders = None
scaler = None

def load_models():
    """Load trained models and preprocessor"""
    global model, preprocessor, conformal_predictors, feature_names, label_encoders, scaler
    
    try:
        # Load the best model (CatBoost)
        model_path = parent_dir / 'models' / 'catboost_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        print(f"✓ Loaded CatBoost model from {model_path}")
        
        # Load conformal predictors (may not exist)
        cp_path = parent_dir / 'models' / 'catboost_conformal.pkl'
        if cp_path.exists():
            conformal_predictors = joblib.load(cp_path)
            print(f"✓ Loaded conformal predictors from {cp_path}")
        else:
            print("⚠ Conformal predictors not found, prediction intervals unavailable")
        
        # Load data to get label encoders
        data_path = parent_dir / 'Data' / 'data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(str(data_path))
        preprocessor.load_data()
        X, y = preprocessor.preprocess()
        
        # Store feature names and encoders
        feature_names = X.columns.tolist()
        label_encoders = preprocessor.label_encoders
        
        # Fit the scaler on all data (since we're using for inference only)
        scaler = StandardScaler()
        scaler.fit(X)
        
        print(f"✓ Loaded preprocessor with {len(feature_names)} features")
        print(f"✓ Available label encoders: {list(label_encoders.keys())}")
        print(f"✓ Scaler fitted on {len(X)} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main page"""
    if model is None:
        return render_template('error.html', 
                             error="Model not loaded. Please check server logs."), 500
    
    # Get unique values for dropdowns
    data_path = parent_dir / 'Data' / 'data.csv'
    df = pd.read_csv(data_path)
    
    dropdown_values = {
        'universities': sorted(df['Universitas'].unique().tolist()),
        'programs': sorted(df['Program'].unique().tolist()),
        'years': sorted(df['Tahun'].unique().tolist()),
        'admissions': sorted(df['Penerimaan'].unique().tolist()),
        'study_programs': sorted(df['Program_Studi'].unique().tolist())
    }
    
    return render_template('index.html', dropdowns=dropdown_values)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['Universitas', 'Program', 'Tahun', 'Penerimaan', 'Program_Studi']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create input DataFrame
        input_df = pd.DataFrame([data])
        
        # Encode categorical features
        for col in required_fields:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError as e:
                    return jsonify({'error': f'Unknown value for {col}: {data[col]}'}), 400
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Get prediction intervals if available
        intervals = None
        if conformal_predictors is not None:
            try:
                lower_bounds = []
                upper_bounds = []
                
                for i, cp in enumerate(conformal_predictors):
                    lower, upper = cp.predict_interval(input_scaled)
                    lower_bounds.append(float(lower[0]))
                    upper_bounds.append(float(upper[0]))
                
                intervals = {
                    'lower': lower_bounds,
                    'upper': upper_bounds
                }
            except Exception as e:
                print(f"Warning: Could not compute intervals: {e}")
        
        # Format response
        ukt_tiers = [f'UKT-{i}' for i in range(1, 12)]
        predictions = {
            tier: {
                'value': float(pred),
                'formatted': f'Rp {pred:,.0f}',
                'lower': intervals['lower'][i] if intervals else None,
                'upper': intervals['upper'][i] if intervals else None,
                'lower_formatted': f'Rp {intervals["lower"][i]:,.0f}' if intervals else None,
                'upper_formatted': f'Rp {intervals["upper"][i]:,.0f}' if intervals else None
            }
            for i, (tier, pred) in enumerate(zip(ukt_tiers, prediction[0]))
        }
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input': data,
            'has_intervals': intervals is not None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    try:
        info = {
            'model_type': 'CatBoost',
            'model_name': 'Gradient Boosted Trees',
            'num_outputs': 11,
            'output_names': [f'UKT-{i}' for i in range(1, 12)],
            'features': feature_names,
            'performance': {
                'R2': 0.9620,
                'RMSE': '1.102M IDR',
                'MAE': '449.5K IDR',
                'MAPE': '7.96%',
                'Coverage': '88.91%'
            },
            'conformal_available': conformal_predictors is not None
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'conformal_loaded': conformal_predictors is not None
    })

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/documentation')
def documentation():
    """Documentation page"""
    return render_template('documentation.html')

if __name__ == '__main__':
    print("=" * 60)
    print("UKT Prediction Web Application")
    print("=" * 60)
    
    # Load models
    if load_models():
        print("\n✓ All models loaded successfully")
        print("\nStarting Flask server...")
        print("Access the application at: http://localhost:5000")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load models. Please check the error messages above.")
        sys.exit(1)
