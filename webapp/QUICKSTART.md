# Quick Start Guide - UKT Prediction Web Application

## 🚀 Start the Web Application

### Windows (PowerShell)

```powershell
# From project root directory
.\.venv\Scripts\python.exe webapp\app.py
```

### Linux/Mac

```bash
# From project root directory
source .venv/bin/activate
python webapp/app.py
```

## 🌐 Access the Application

Once the server starts, you'll see:
```
✓ All models loaded successfully
Starting Flask server...
Access the application at: http://localhost:5000
```

Open your browser and navigate to:
- **Main App**: http://localhost:5000
- **About**: http://localhost:5000/about
- **Documentation**: http://localhost:5000/documentation

## 📝 Using the Application

1. **Select University**: Choose from 12 Indonesian public universities
2. **Select Program Level**: D3, D4, or S1
3. **Select Academic Year**: 2025/2026 (or available years)
4. **Select Admission Method**: SNBP/SNBT
5. **Search Study Program**: Type to filter from 378+ programs
6. **Click Predict**: Get predictions for all 11 UKT tiers

## 🎯 Features

✅ **Predictions**: All 11 UKT tiers (UKT-1 to UKT-11)
✅ **Confidence Intervals**: 90% prediction intervals (when available)
✅ **Visualizations**: Interactive Chart.js charts
✅ **Responsive Design**: Works on desktop, tablet, and mobile
✅ **RESTful API**: JSON endpoints for programmatic access

## 🔌 API Endpoints

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Model Info
```bash
curl http://localhost:5000/api/model-info
```

### Prediction
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

## ⚠️ Requirements

- Python 3.11+
- Flask 3.1.0
- Trained models in `models/` directory
- Dataset in `Data/data.csv`

## 🛑 Stopping the Server

Press `CTRL+C` in the terminal where the server is running.

## 📞 Troubleshooting

### Model not loaded
```bash
# Train models first
python src/main.py
```

### Port already in use
Edit `webapp/app.py` and change:
```python
app.run(debug=True, host='0.0.0.0', port=8000)  # Change port number
```

### Import errors
```bash
# Install Flask
pip install flask

# Or install all dependencies
pip install -r webapp/requirements.txt
```

## 🎨 Customization

### Change Port
Edit `webapp/app.py`, line ~200:
```python
app.run(debug=True, host='0.0.0.0', port=YOUR_PORT)
```

### Disable Debug Mode
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

### Production Deployment
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:5000 --chdir webapp app:app
```

## 📚 More Information

- **Full README**: `webapp/README.md`
- **Research Docs**: `docs/RESEARCH_DOCUMENTATION.md`
- **GitHub**: https://github.com/lyonad/RF-vs-GB-for-UKT-PTN-Prediction

---

**Built with ❤️ for Indonesian Higher Education**
