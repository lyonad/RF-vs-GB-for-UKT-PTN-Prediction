# Configuration file for Indonesian Public University Tuition Fees Prediction Project
# config.py

# Data configuration
DATA_PATH = "Data/data.csv"
PROCESSED_DATA_PATH = "Data/processed_data.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
SIGNIFICANCE_LEVEL = 0.1  # For conformal prediction (90% confidence interval)

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

CATBOOST_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'verbose': False
}

LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'verbose': -1
}

# Quantile regression configuration
QUANTILES = [0.1, 0.5, 0.9]  # 10th, 50th (median), and 90th percentiles

# Evaluation metrics
EVALUATION_METRICS = [
    'MSE',
    'RMSE', 
    'MAE',
    'R2',
    'Pearson_Corr',
    'Coverage'  # For conformal prediction
]

# Output paths
MODELS_OUTPUT_PATH = "models/"
RESULTS_OUTPUT_PATH = "results/"
PLOTS_OUTPUT_PATH = "results/plots/"

# Target variables (tuition fees for different years)
TARGET_COLUMNS = ['UKT-1', 'UKT-2', 'UKT-3', 'UKT-4']

# Feature columns (to be updated after preprocessing)
FEATURE_COLUMNS = [
    'university_type',
    'location', 
    'accreditation',
    'program_type',
    'faculty',
    'urban_rural',
    'region_development',
    'student_capacity',
    'years_since_established'
]