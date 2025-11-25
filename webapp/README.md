# UKT Prediction Web Application

Professional web interface for the UKT (Uang Kuliah Tunggal) Prediction System - Indonesian Public University Tuition Fee Predictor.

## Features

- **Interactive Prediction Interface**: User-friendly form for inputting university and program details
- **Real-time Predictions**: Get instant predictions for all 11 UKT tiers
- **Confidence Intervals**: 90% prediction intervals using conformal prediction
- **Interactive Visualizations**: Chart.js powered visualizations of predictions
- **Responsive Design**: Modern, mobile-friendly UI
- **RESTful API**: JSON API endpoints for programmatic access
- **Professional Styling**: Clean, modern design with Inter font family

## Quick Start

### Prerequisites

- Python 3.11+
- Trained models (run `python src/main.py` from project root if not already trained)

### Installation

```bash
# 1. Navigate to webapp directory
cd webapp

# 2. Install dependencies (if not installed globally)
pip install -r requirements.txt

# 3. Run the application
python app.py

# 4. Open your browser
# Navigate to: http://localhost:5000
```

## Project Structure

```
webapp/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── static/
│   ├── css/
│   │   └── style.css      # Main stylesheet
│   └── js/
│       └── main.js        # Frontend JavaScript
├── templates/
│   ├── index.html         # Main prediction page
│   ├── about.html         # About page
│   └── documentation.html # Documentation page
└── api/
    └── (future API modules)
```

## API Endpoints

### POST /api/predict

Predict UKT fees for given input features.

**Request:**
```json
{
  "Universitas": "UGM",
  "Program": "S1",
  "Tahun": "2025/2026",
  "Penerimaan": "SNBP/SNBT",
  "Program_Studi": "Teknik Informatika"
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "UKT-1": {
      "value": 500000,
      "formatted": "Rp 500,000",
      "lower": 450000,
      "upper": 550000
    },
    ...
  },
  "has_intervals": true
}
```

### GET /api/model-info

Get model information and performance metrics.

### GET /api/health

Health check endpoint.

## Features in Detail

### 1. Smart Study Program Search

- Type-ahead search for 378+ study programs
- Real-time filtering as you type
- Easy selection from filtered results

### 2. Prediction Results

- Point predictions for all 11 UKT tiers
- 90% confidence intervals with lower and upper bounds
- Color-coded result cards with hover effects
- Input summary for verification

### 3. Interactive Charts

- Line chart showing predictions across all tiers
- Confidence interval visualization (shaded area)
- Responsive and interactive with Chart.js
- Tooltips with formatted currency values

### 4. Professional UI/UX

- Modern gradient design
- Smooth animations and transitions
- Mobile-responsive layout
- Inter font family for clean typography
- Accessible form controls

## Technology Stack

- **Backend**: Flask 3.1.0
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Charts**: Chart.js 4.x
- **Fonts**: Google Fonts (Inter)
- **ML Framework**: scikit-learn, CatBoost
- **Data Processing**: pandas, numpy

## Model Information

- **Model Type**: CatBoost (Gradient Boosted Trees)
- **Performance**: R² = 0.9620, MAPE = 7.96%
- **Outputs**: 11 UKT tiers (UKT-1 to UKT-11)
- **Uncertainty**: Conformal prediction intervals

## Configuration

### Environment Variables

You can customize the application using environment variables:

```bash
# Flask configuration
export FLASK_ENV=development
export FLASK_DEBUG=1
export FLASK_SECRET_KEY=your-secret-key

# Server configuration
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
```

### Model Paths

The application looks for trained models in:
- `../models/catboost_best.pkl` (main model)
- `../models/catboost_conformal.pkl` (conformal predictors)
- `../Data/data.csv` (training data for label encoders)

## Testing

Test the API endpoints:

```bash
# Health check
curl http://localhost:5000/api/health

# Model info
curl http://localhost:5000/api/model-info

# Prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Universitas": "UGM",
    "Program": "S1",
    "Tahun": "2025/2026",
    "Penerimaan": "SNBP/SNBT",
    "Program_Studi": "Teknik Informatika"
  }'
```

## Deployment

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "webapp.app:app"]
```

## Documentation

- **User Guide**: Visit `/documentation` in the web app
- **About**: Visit `/about` for project information
- **Full Research**: See `docs/RESEARCH_DOCUMENTATION.md` in project root
- **GitHub**: [RF-vs-GB-for-UKT-PTN-Prediction](https://github.com/lyonad/RF-vs-GB-for-UKT-PTN-Prediction)

## Troubleshooting

### Common Issues

**Model not loaded:**
```bash
# Train models first
cd ..
python src/main.py
cd webapp
python app.py
```

**Import errors:**
```bash
# Install all dependencies
pip install -r ../requirements.txt
pip install -r requirements.txt
```

**Port already in use:**
```bash
# Use a different port
python app.py --port 8000
# Or modify app.py: app.run(port=8000)
```

## License

This project is licensed under the MIT License. See LICENSE file in the project root.

## Author

**Lyon Ambrosio Djuanda**

## Acknowledgments

- **Dataset**: Irvi Aini - UKT PTN Indonesia (Kaggle)
- **ML Libraries**: scikit-learn, CatBoost, XGBoost, LightGBM
- **Web Framework**: Flask
- **Visualization**: Chart.js

## Support

- **GitHub**: [Issues](https://github.com/lyonad/RF-vs-GB-for-UKT-PTN-Prediction/issues)
- **DOI**: [10.5281/zenodo.17504815](https://doi.org/10.5281/zenodo.17504815)

---

**Built for Indonesian Higher Education**
