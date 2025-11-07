# UKT Prediction Web Application - Complete Guide

A professional, modern web interface for predicting Indonesian public university tuition fees (UKT) across all 11 tiers using state-of-the-art machine learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Features](#features)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Frontend Architecture](#frontend-architecture)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This web application provides:
- **Interactive UI** for UKT predictions with form-based input
- **Real-time predictions** for all 11 UKT tiers (UKT-1 to UKT-11)
- **90% confidence intervals** using conformal prediction
- **RESTful API** endpoints for programmatic access
- **Responsive design** that works on desktop, tablet, and mobile
- **Professional styling** with modern design tokens and consistent UI components

### Key Statistics
- **96.2%** R² Score Accuracy
- **7.96%** Mean Error Rate
- **88.91%** Confidence Coverage
- **CatBoost** ML Algorithm

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Trained models in `../models/` directory
- Dataset in `../Data/data.csv`

### Running Locally

**Windows (PowerShell):**
```powershell
# From project root
.\.venv\Scripts\python.exe webapp\app.py
```

**Linux/Mac:**
```bash
# From project root
source .venv/bin/activate
python webapp/app.py
```

**Access the application:**
- Main UI: http://localhost:5000
- About Page: http://localhost:5000/about
- API Docs: http://localhost:5000/documentation

---

## 🌟 Features

### 1. Intelligent Prediction Form

- **University Selection**: 12 Indonesian public universities
- **Program Levels**: D3 (Diploma), D4 (Applied Bachelor), S1 (Bachelor)
- **Academic Year**: 2025/2026 and available years
- **Admission Method**: SNBP/SNBT selection paths
- **Smart Study Program Search**: 
  - Type-ahead filtering across 378+ programs
  - Real-time search as you type
  - Instant selection from filtered results

### 2. Comprehensive Results Display

- **Point Predictions**: For all 11 UKT tiers
- **Confidence Intervals**: 90% lower and upper bounds (when available)
- **Input Summary**: Verification of submitted data
- **Interactive Chart**: 
  - Line chart with predictions across tiers
  - Shaded confidence interval regions
  - Hover tooltips with formatted currency
  - Responsive Chart.js visualization

### 3. Professional UI/UX

- **Modern Design System**:
  - Design tokens for consistent spacing, colors, and typography
  - Navy blue primary color (#001F3F) for trust and stability
  - Teal accent color (#00A896) for CTAs and highlights
  - Playfair Display for headlines, Montserrat for subheadings, Roboto for body
- **Sectioned Layout**:
  - On-page navigation: Overview | Predict | Results | Insights | FAQ
  - Smooth scrolling and scrollspy for active section highlighting
  - Clear visual hierarchy with semantic HTML5 sections
- **Responsive Breakpoints**:
  - Desktop (1280px max-width)
  - Tablet (1024px and below)
  - Mobile (768px and below)
- **Interactive Elements**:
  - Smooth animations and transitions
  - Hover effects on cards and buttons
  - Global alert banner for non-blocking feedback
  - API status indicator in navbar

### 4. FAQ Section

- What are UKT tiers?
- How accurate are the predictions?
- Do I need all inputs?
- Why don't I see intervals?

### 5. RESTful API

- JSON endpoints for external integration
- Health check and model info endpoints
- CORS support for cross-origin requests

---

## 📁 Project Structure

```
webapp/
├── app.py                     # Flask application entry point
├── requirements.txt           # Python dependencies (Flask, etc.)
├── README.md                  # Original README
├── QUICKSTART.md             # Quick start guide
├── PROJECT_SUMMARY.md        # Project overview
├── WEBAPP_GUIDE.md           # This comprehensive guide
├── static/
│   ├── css/
│   │   └── style.css         # Main stylesheet (design system, sections, responsive)
│   └── js/
│       └── main.js           # Frontend logic (form, charts, scrollspy, alerts)
├── templates/
│   ├── base.html             # Shared layout (navbar, footer, alert banner, scroll-to-top)
│   ├── index.html            # Main prediction page (sectioned: hero, overview, predict, results, insights, FAQ)
│   ├── about.html            # About the project
│   ├── documentation.html    # API documentation
│   └── error.html            # Error page
└── api/
    └── (future modular API routes)
```

---

## 🔌 API Documentation

### Endpoints

#### `POST /api/predict`

Predict UKT fees for all 11 tiers.

**Request Body:**
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
      "upper": 550000,
      "lower_formatted": "Rp 450,000",
      "upper_formatted": "Rp 550,000"
    },
    "UKT-2": { ... },
    ...
    "UKT-11": { ... }
  },
  "has_intervals": true
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Missing required field: Universitas"
}
```

#### `GET /api/model-info`

Retrieve model metadata and performance metrics.

**Response:**
```json
{
  "model_type": "CatBoostRegressor",
  "model_name": "CatBoost Multi-Output Regressor",
  "accuracy": {
    "r2_score": 0.9620,
    "mape": 7.96
  },
  "outputs": 11,
  "conformal_available": true
}
```

#### `GET /api/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-07T12:34:56Z"
}
```

### Testing with cURL

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

---

## 🎨 Frontend Architecture

### Design System (CSS)

**Color Palette:**
- Primary: `#001F3F` (Navy Blue) - Trust & Stability
- Accent: `#00A896` (Teal) - Action & Energy
- Background: `#F5F5F5` (Off-white)
- Surface: `#FFFFFF`
- Text: `#001F3F` (primary), `#4a5568` (secondary)

**Typography:**
- Headlines: Playfair Display (serif)
- Subheadings: Montserrat (sans-serif)
- Body: Roboto (sans-serif)

**Spacing Tokens:**
- xs: 8px, sm: 16px, md: 24px, lg: 32px, xl: 48px, 2xl: 64px

**Components:**
- Cards with shadow and hover effects
- Buttons with gradient and animation
- Form inputs with focus states
- Stat cards with icons and hover highlights
- Global alert banner (info, warning, error)
- API status pill in navbar

### JavaScript Features (`main.js`)

1. **Form Handling**:
   - Real-time validation
   - Loading states
   - Error handling with global alerts

2. **Study Program Search**:
   - Type-ahead filtering
   - Instant selection

3. **Results Display**:
   - Dynamic results grid generation
   - Chart.js integration
   - Smooth scroll to results

4. **Scrollspy & Navigation**:
   - On-page section navigation with smooth scroll
   - Active section highlighting via IntersectionObserver
   - Scroll-to-top button

5. **Health Check**:
   - API status indicator in navbar
   - Auto-check on page load

### Template Architecture

**`base.html`**: Shared layout
- Navbar with active state highlighting
- Global alert banner
- API status pill
- Footer
- Scroll-to-top button

**`index.html`**: Main page (extends `base.html`)
- Sectioned structure: hero, overview (stats), predict (form), results, insights (chart), FAQ
- On-page navigation for quick jumps
- Chart.js included via block

**Other pages** (`about.html`, `documentation.html`, `error.html`): Extend `base.html` for consistency

---

## 🚢 Deployment

### Development Mode

```bash
python app.py
# Flask dev server on http://localhost:5000
```

### Production with Gunicorn

```bash
# Install Gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "webapp.app:app"]
```

**Build and Run:**
```bash
docker build -t ukt-predictor .
docker run -p 5000:5000 ukt-predictor
```

### Environment Variables

```bash
# Flask configuration
export FLASK_ENV=production
export FLASK_DEBUG=0
export FLASK_SECRET_KEY=your-secret-key

# Server
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
```

---

## 🐛 Troubleshooting

### Model Not Loaded

**Symptoms:**
- 500 errors when accessing `/api/predict`
- "Model not loaded" message

**Solution:**
```bash
# Train models first
cd ..
python src/main.py
cd webapp
python app.py
```

**Verify files exist:**
- `models/catboost_model.pkl`
- `results/conformal_predictors.pkl`
- `Data/data.csv`

### Import Errors

**Symptoms:**
- `ModuleNotFoundError: No module named 'flask'`

**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt
```

### Port Already in Use

**Symptoms:**
- `OSError: [Errno 48] Address already in use`

**Solution:**
```python
# Edit app.py, change port:
app.run(debug=True, host='0.0.0.0', port=8000)
```

Or kill the existing process:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <pid> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### Horizontal Scrollbar

**Symptoms:**
- Page scrolls left-right on desktop

**Solution:**
- Already fixed in `style.css`:
  - `body { overflow-x: hidden; }`
  - `.hero-section::before { width: 100%; }` (not `100vw`)
  - `.page-nav { max-width: 100%; flex-wrap: wrap; }`
  - `img, canvas, svg, video { max-width: 100%; }`

### Conformal Intervals Not Showing

**Symptoms:**
- Predictions display, but no confidence intervals

**Possible Causes:**
1. `results/conformal_predictors.pkl` missing or incorrect format
2. Conformal predictor not available for current model

**Solution:**
- Re-run `src/main.py` to generate conformal predictors
- Check `app.py` for correct path to conformal file

---

## 📚 Additional Resources

- **Research Documentation**: `../docs/RESEARCH_DOCUMENTATION.md`
- **Project Overview**: `../docs/PROJECT_OVERVIEW.md`
- **GitHub Repository**: [RF-vs-GB-for-UKT-PTN-Prediction](https://github.com/lyonad/RF-vs-GB-for-UKT-PTN-Prediction)
- **DOI**: [10.5281/zenodo.17504815](https://doi.org/10.5281/zenodo.17504815)
- **Dataset**: [UKT PTN Indonesia (Kaggle)](https://www.kaggle.com/datasets/irvifa/ukt-ptn-indonesia-s1-d4-d3)

---

## 🛠️ Technology Stack

- **Backend**: Flask 3.1.0
- **ML**: scikit-learn, CatBoost, XGBoost, LightGBM
- **Data**: pandas, numpy
- **Frontend**: HTML5, CSS3 (Design Tokens), JavaScript ES6+
- **Charts**: Chart.js 4.x
- **Fonts**: Google Fonts (Playfair Display, Montserrat, Roboto)

---

## 📄 License

MIT License - See `../LICENSE` in project root.

---

## 👨‍💻 Author

**Lyon Ambrosio Djuanda**

---

## 🙏 Acknowledgments

- **Dataset**: Irvi Aini (Kaggle)
- **ML Libraries**: scikit-learn, CatBoost, XGBoost, LightGBM
- **Web Framework**: Flask
- **Visualization**: Chart.js, matplotlib, plotly

---

**Built with ❤️ for Indonesian Higher Education**
