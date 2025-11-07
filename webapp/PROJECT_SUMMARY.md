# 🎓 UKT Prediction Web Application - Project Summary

## ✅ Successfully Created

A complete, professional web application for predicting Indonesian Public University tuition fees (UKT) has been created in the `webapp/` directory.

## 📁 Project Structure

```
webapp/
├── app.py                     # Flask backend server (206 lines)
├── requirements.txt           # Dependencies (Flask, Werkzeug, Jinja2)
├── README.md                  # Complete documentation
├── QUICKSTART.md             # Quick start guide
├── .gitignore                # Git ignore rules
├── static/
│   ├── css/
│   │   └── style.css         # Modern responsive styling (700+ lines)
│   └── js/
│       └── main.js           # Frontend JavaScript with Chart.js
├── templates/
│   ├── index.html            # Main prediction page
│   ├── about.html            # About page
│   ├── documentation.html    # Documentation page
│   └── error.html            # Error page
└── api/
    └── (ready for future expansion)
```

## 🌟 Key Features

### 1. **Professional UI/UX**
- ✅ Modern gradient design with Inter font
- ✅ Fully responsive (mobile, tablet, desktop)
- ✅ Smooth animations and transitions
- ✅ Intuitive form with smart search
- ✅ Interactive visualizations

### 2. **Backend API**
- ✅ Flask 3.1.0 server
- ✅ RESTful JSON endpoints
- ✅ Model loading and preprocessing
- ✅ Error handling and validation
- ✅ Health check endpoint

### 3. **Prediction System**
- ✅ CatBoost model integration
- ✅ All 11 UKT tiers predicted
- ✅ 90% confidence intervals (when available)
- ✅ Real-time results display
- ✅ Interactive Chart.js visualizations

### 4. **Documentation**
- ✅ Complete README with setup instructions
- ✅ Quick start guide
- ✅ API documentation
- ✅ Troubleshooting section
- ✅ About and Documentation pages

## 🚀 Quick Start

### Start the Server
```powershell
# Windows PowerShell (from project root)
.\.venv\Scripts\python.exe webapp\app.py
```

### Access the Application
```
http://localhost:5000
```

## 🎯 Usage

1. **Select Input Features**
   - University (12 options)
   - Program Level (D3/D4/S1)
   - Academic Year (2025/2026)
   - Admission Method (SNBP/SNBT)
   - Study Program (378+ options with search)

2. **Get Predictions**
   - Click "Predict Tuition Fees"
   - View results for all 11 UKT tiers
   - See confidence intervals
   - Explore interactive chart

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main prediction interface |
| `/about` | GET | About page |
| `/documentation` | GET | Documentation page |
| `/api/predict` | POST | Get predictions (JSON) |
| `/api/model-info` | GET | Model information |
| `/api/health` | GET | Health check |

### Example API Call
```bash
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

## 📊 Model Performance

The web app uses the CatBoost model, which achieved:
- **R² Score**: 0.9620 (96.2% variance explained)
- **RMSE**: 1.102M IDR
- **MAE**: 449.5K IDR
- **MAPE**: 7.96%
- **Coverage**: 88.91% (prediction intervals)

## 🎨 Design Highlights

### Color Scheme
- Primary: #2563eb (blue)
- Secondary: #10b981 (green)
- Background: #f8fafc (light gray)
- Surface: #ffffff (white)
- Text: #1e293b (dark)

### Typography
- Font: Inter (Google Fonts)
- Weights: 300, 400, 500, 600, 700

### Components
- Modern gradient buttons
- Card-based layout
- Responsive grid system
- Smooth hover effects
- Chart.js visualizations

## 🛠️ Technology Stack

### Backend
- Flask 3.1.0
- Python 3.11+
- scikit-learn
- CatBoost
- pandas, numpy

### Frontend
- HTML5
- CSS3 (Custom, no frameworks)
- JavaScript (ES6+)
- Chart.js 4.x
- Google Fonts (Inter)

### Features
- RESTful API
- JSON responses
- Form validation
- Error handling
- Responsive design

## ✅ Testing Results

**Server Status**: ✓ Running successfully
**URL**: http://localhost:5000
**Model**: ✓ Loaded (CatBoost)
**Preprocessor**: ✓ Loaded (5 features, 378 study programs)
**Label Encoders**: ✓ Available for all categorical features

**Note**: Conformal predictors not available (optional feature)

## 🚢 Deployment Options

### Development
```bash
python webapp/app.py
```

### Production
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:5000 --chdir webapp app:app
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "webapp.app:app"]
```

## 📚 Documentation

All documentation is included:
- `webapp/README.md`: Full documentation
- `webapp/QUICKSTART.md`: Quick start guide
- `/about`: About page in web app
- `/documentation`: Documentation page in web app

## 🎉 What You Get

✅ **Complete Web Application**: Fully functional, ready to use
✅ **Professional Design**: Modern, responsive UI
✅ **RESTful API**: JSON endpoints for integration
✅ **Documentation**: Complete guides and examples
✅ **Error Handling**: Graceful error pages and messages
✅ **Extensible**: Easy to add new features

## 🔧 Customization

The application is designed to be easily customizable:
- **Styling**: Edit `static/css/style.css`
- **Behavior**: Edit `static/js/main.js`
- **Backend**: Edit `app.py`
- **Templates**: Edit files in `templates/`

## 📞 Support

- **GitHub**: https://github.com/lyonad/RF-vs-GB-for-UKT-PTN-Prediction
- **DOI**: 10.5281/zenodo.17504815
- **Issues**: GitHub Issues page

## 🙏 Credits

- **Author**: Lyon Ambrosio Djuanda
- **Dataset**: Irvi Aini (Kaggle)
- **License**: MIT

---

**The web application is now ready to use!** 🎉

Start the server with:
```powershell
.\.venv\Scripts\python.exe webapp\app.py
```

Then open http://localhost:5000 in your browser.
