# Comparative Study of Random Forest and Gradient-Boosted Trees for Predicting Indonesian Public University Tuition Fees with Multi-Output Quantile and Conformal Prediction

**Author**: Lyon Ambrosio Djuanda  
**Year**: 2025  
**License**: MIT  
**DOI**: (to be updated)

For citation formats (BibTeX, EndNote), see `CITATION.cff`.

## Project Overview

This research project conducts a comprehensive comparative study of Random Forest and Gradient-Boosted Trees (XGBoost, CatBoost, and LightGBM) for predicting Indonesian public university tuition fees (UKT - Uang Kuliah Tunggal) across 11 fee categories with advanced uncertainty quantification. The study implements multi-output quantile regression and conformal prediction to provide both point estimates and reliable prediction intervals.

**Key Results**: CatBoost achieves the best predictive accuracy (R² = 0.9620, MAPE = 7.96%) with 88.91% interval coverage, while all models demonstrate excellent performance (R² > 0.94).

## Research Objectives

- Compare the performance of Random Forest and Gradient-Boosted Trees for multi-output tuition fee prediction
- Implement multi-output quantile regression for uncertainty quantification
- Apply conformal prediction methods for reliable prediction intervals
- Evaluate models on accuracy, reliability, and uncertainty quantification
- Analyze feature importance and provide insights into tuition fee determination factors

## Dataset

**Source**: [UKT PTN Indonesia - S1, D4, D3](https://www.kaggle.com/datasets/irvifa/ukt-ptn-indonesia-s1-d4-d3) by Irvi Aini (Kaggle)  
**License**: MIT License

The dataset contains Indonesian public university (PTN) tuition fee information with:

- **Universitas**: University name (PTN)
- **Program**: Degree program (S1, D4, D3)
- **Tahun**: Academic year (2025/2026)
- **Penerimaan**: Admission path (SNBP or SNBT)
- **Program_Studi**: Study program name
- **UKT-1 to UKT-11**: Tuition fees for each UKT level in IDR

## Methodology

### 1. Data Preprocessing
- Data loading and cleaning
- Categorical variable encoding
- Feature scaling
- Train/test split

### 2. Model Implementation
- Random Forest with multi-output support
- XGBoost with multi-output support
- CatBoost with multi-output support
- LightGBM with multi-output support

### 3. Advanced Techniques
- **Multi-Output Quantile Regression**: Predicts 10th, 50th, and 90th percentiles
- **Conformal Prediction**: Distribution-free prediction intervals with guaranteed coverage
- **Shared Calibration**: Methodological innovation ensuring fairness across all 11 outputs

### 4. Model Evaluation
- RMSE, MAE, R², Pearson Correlation
- Prediction interval coverage
- Feature importance analysis

## Project Structure

```
UKT-PTN/
├── README.md
├── requirements.txt
├── setup.bat
├── run_research.bat
├── .gitignore
├── Data/
│   └── data.csv              # Original dataset
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── data_preprocessor.py  # Data preprocessing module
│   ├── models.py             # ML models implementation
│   ├── conformal_prediction.py # Conformal prediction module
│   ├── evaluation.py         # Model evaluation module
│   └── main.py               # Main execution script
├── utils/
│   ├── __init__.py
│   └── visualization.py      # Visualization utilities
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── models/                   # Trained models saved here
├── results/                  # Analysis results saved here
├── docs/                     # Documentation files
└── tests/                    # Unit and integration tests
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/lyonad/RF-vs-GB-for-UKT-PTN-Prediction.git
   cd RF-vs-GB-for-UKT-PTN-Prediction
   ```

2. Run the setup script to create the virtual environment and install dependencies:
   ```bash
   setup.bat
   ```
   
   This will:
   - Create a virtual environment (.venv)
   - Activate the virtual environment
   - Install all required packages from requirements.txt

3. Alternatively, you can manually create the virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Running the Main Analysis

Execute the complete research pipeline with a single command:
```bash
run_research.bat
```

Or run manually:
```bash
python src/main.py
```

### Jupyter Notebook Analysis

To perform exploratory data analysis:
```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

## Key Features

1. **Multi-Output Prediction**: Models predict tuition fees for multiple UKT categories simultaneously
2. **Uncertainty Quantification**: Implementation of conformal prediction for reliable confidence intervals
3. **Quantile Regression**: Provides prediction intervals using quantile regression techniques
4. **Comprehensive Evaluation**: Multiple metrics for model comparison
5. **Visualization Tools**: Rich set of plots for model analysis and interpretation
6. **Automated Reporting**: Generates detailed research reports

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Measures the average prediction error
- **MAE (Mean Absolute Error)**: Measures the average absolute prediction error
- **R² (Coefficient of Determination)**: Indicates the proportion of variance explained
- **Pearson Correlation**: Measures the linear correlation between predictions and actual values
- **Prediction Interval Coverage**: Measures the percentage of actual values within predicted intervals
- **MAPE (Mean Absolute Percentage Error)**: Measures relative prediction accuracy

## Algorithms Compared

1. **Random Forest**: Ensemble method using multiple decision trees
2. **XGBoost**: Extreme Gradient Boosting with regularization
3. **CatBoost**: Gradient boosting with categorical feature handling
4. **LightGBM**: Light Gradient Boosting Machine with efficient implementation

## Advanced Techniques

### Multi-Output Quantile Regression
Extends traditional quantile regression to predict multiple outputs simultaneously, providing uncertainty estimates for each target variable.

### Conformal Prediction
A framework for producing prediction sets that are valid in the sense of having a guaranteed coverage probability, independent of the data distribution.

## Results

### Model Performance (Overall Metrics)

| Rank | Model | R² | RMSE (IDR) | MAE (IDR) | MAPE (%) | Coverage (%) |
|------|-------|-----|------------|-----------|----------|--------------|
| 1 | **CatBoost** | **0.9620** | **1,102,068** | **449,465** | **7.96** | 88.91 |
| 2 | LightGBM | 0.9574 | 1,166,708 | 519,485 | 8.95 | 89.30 |
| 3 | RandomForest | 0.9514 | 1,246,322 | 448,081 | 7.55 | **91.10** |
| 4 | XGBoost | 0.9417 | 1,365,083 | 489,653 | 8.58 | 89.75 |

**Key Findings**:
- **Best Point Accuracy**: CatBoost (R² = 0.9620, explains 96.2% of variance)
- **Best Interval Coverage**: RandomForest (91.10%, exceeds 90% target)
- **All models**: R² > 0.94, MAPE < 9% (excellent performance)
- **Practical Accuracy**: 77.76% of CatBoost predictions within 10% of true value

### Deliverables
- ✅ Comparative analysis of 4 tree-based models with fair comparison protocol
- ✅ Complete performance metrics (MSE, RMSE, MAE, R², MAPE, Pearson, Coverage)
- ✅ Uncertainty quantification via quantile regression and conformal prediction
- ✅ Feature importance analysis for all models
- ✅ Trained models saved in `models/` directory
- ✅ Comprehensive research report in `results/research_report.md`
- ✅ Interactive visualizations in `results/visualizations/`
- ✅ **Publication-ready documentation in `docs/RESEARCH_DOCUMENTATION.md`**

## Documentation

### Research Documentation

**Complete academic documentation** available in `docs/RESEARCH_DOCUMENTATION.md` covering:
- Introduction, Literature Review, and Research Questions
- Detailed Methodology (data, preprocessing, models, evaluation)
- Results with statistical analysis
- Discussion and Conclusion
- Limitations and Future Work
- References and Appendices

**Suitable for**: Scopus article preparation, thesis chapters, technical reports

### Project Overview

**Technical documentation** available in `docs/PROJECT_OVERVIEW.md` with:
- Code-level architecture and module descriptions
- How to run, configure, and extend the project
- Development tips and edge case handling

## Testing

Run the test suite to ensure all components work correctly:
```bash
python -m pytest tests/test_project.py -v
```

**Test Results**: 12 tests passed, full coverage of core components

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{djuanda_2025_ukt_prediction,
  author = {Djuanda, Lyon Ambrosio},
  title={Comparative Study of Random Forest and Gradient-Boosted Trees for Predicting Indonesian Public University Tuition Fees with Multi-Output Quantile and Conformal Prediction},
   year={2025},
    publisher = {Zenodo},
   note={Research project with CatBoost achieving best performance for multi-output UKT prediction}
}
```

**Dataset Citation**:
```bibtex
@dataset{aini_2025_ukt,
  author = {Aini, Irvi},
  title = {UKT PTN Indonesia - S1, D4, D3},
   year = {2025},
  publisher = {Kaggle},
   url = {https://www.kaggle.com/datasets/irvifa/ukt-ptn-indonesia-s1-d4-d3},
  note = {MIT License}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Dataset**: Irvi Aini for providing the UKT PTN Indonesia dataset on Kaggle
- **Software**: scikit-learn, XGBoost, CatBoost, LightGBM, and Python scientific computing community
- **Indonesian Ministry of Education, Culture, Research, and Technology** for UKT policy framework