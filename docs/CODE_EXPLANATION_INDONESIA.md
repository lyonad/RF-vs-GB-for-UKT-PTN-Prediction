# Penjelasan Menyeluruh Cara Kerja Kode Proyek UKT-PTN Prediction

## Ringkasan Proyek

Proyek ini adalah sistem prediksi biaya kuliah (UKT - Uang Kuliah Tunggal) untuk Perguruan Tinggi Negeri (PTN) di Indonesia menggunakan berbagai algoritma machine learning. Proyek membandingkan 4 model: Random Forest, XGBoost, CatBoost, dan LightGBM, dengan teknik advanced seperti quantile regression dan conformal prediction untuk uncertainty quantification.

---

## Arsitektur Proyek

Proyek ini mengikuti struktur modular yang terorganisir dengan baik:

```
UKT-PTN/
├── src/                    # Kode sumber utama
├── utils/                  # Utility functions
├── webapp/                 # Aplikasi web Flask
├── Data/                   # Dataset
├── models/                 # Model yang sudah dilatih
├── results/                # Hasil analisis dan visualisasi
├── tests/                  # Unit tests
└── docs/                   # Dokumentasi
```

---

## Penjelasan Modul per Modul

### 1. **src/config.py** - Konfigurasi Proyek

**Fungsi**: Menyimpan semua pengaturan konfigurasi proyek

**Cara Kerja**:
- Menyimpan path ke file data (`DATA_PATH`)
- Mengatur parameter model (Random Forest, XGBoost, CatBoost, LightGBM)
- Menentukan quantile yang digunakan (`[0.1, 0.5, 0.9]` untuk 10%, 50%, 90%)
- Mengatur tingkat signifikansi untuk conformal prediction (0.1 = 90% confidence interval)
- Menyimpan daftar metrik evaluasi dan path output

**Konteks Penggunaan**: Diimport oleh modul lain untuk mengakses konfigurasi tanpa hardcoding

---

### 2. **src/data_preprocessor.py** - Preprocessing Data

**Fungsi**: Menangani loading, cleaning, dan preprocessing data sebelum training

#### **Kelas: `DataPreprocessor`**

**Metode Utama**:

1. **`__init__(data_path)`**
   - Inisialisasi dengan path ke file CSV
   - Membuat instance `StandardScaler` dan dictionary untuk `label_encoders`

2. **`load_data()`**
   - Membaca file CSV menggunakan pandas
   - Jika file tidak ditemukan, membuat sample data untuk demonstrasi
   - Mengembalikan DataFrame

3. **`_create_sample_data()`** (private method)
   - Membuat dataset sintetis jika data asli tidak tersedia
   - Membuat fitur seperti: universitas, lokasi, akreditasi, program, fakultas
   - Membuat kolom UKT-1 hingga UKT-11 dengan pola yang realistis
   - Menyimpan ke CSV untuk penggunaan selanjutnya

4. **`preprocess(target_columns=None)`**
   - Memisahkan fitur (X) dan target (y)
   - **Encoding Kategori**: Menggunakan `LabelEncoder` untuk mengkonversi kategorikal menjadi numerik
     - Contoh: "UGM" → 0, "ITB" → 1, "UI" → 2
   - **Handling Missing Values**: Mengisi NaN dengan mean
   - Mengembalikan tuple (X, y)

5. **`split_data(X, y, test_size=0.2)`**
   - Membagi data menjadi training (80%) dan test (20%)
   - Menggunakan `train_test_split` dari scikit-learn
   - Mengembalikan (X_train, X_test, y_train, y_test)

6. **`scale_features(X_train, X_test)`**
   - **Standardisasi**: Menggunakan `StandardScaler`
   - Formula: `z = (x - μ) / σ`
   - Fit pada training data, transform pada training dan test
   - Penting untuk algoritma yang sensitif terhadap skala (seperti SVM, neural networks)

**Alur Preprocessing**:
```
CSV File → Load → Identify UKT columns → Separate Features/Targets → 
Encode Categorical → Fill Missing Values → Split Train/Test → Scale Features
```

---

### 3. **src/models.py** - Implementasi Model Machine Learning

**Fungsi**: Mengimplementasikan semua model ML dan quantile regression

#### **Kelas: `MultiOutputQuantileRegressor`**

**Fungsi**: Wrapper untuk quantile regression yang mendukung multi-output

**Cara Kerja**:
- Untuk setiap quantile (0.1, 0.5, 0.9), melatih model terpisah
- **XGBoost**: Menggunakan `objective='reg:quantileerror'` dengan `quantile_alpha`
- **CatBoost**: Menggunakan `loss_function='Quantile:alpha={q}'`
- **LightGBM**: Menggunakan `objective='quantile'` dengan `alpha=q`
- **Random Forest**: Tidak memiliki native quantile support, menggunakan pendekatan standar
- Setiap model dibungkus dengan `MultiOutputRegressor` untuk prediksi multi-output

**Contoh**: Untuk memprediksi UKT-1 hingga UKT-11 dengan 3 quantiles, akan ada 3 model yang dilatih, masing-masing menghasilkan 11 prediksi.

#### **Kelas: `TuitionFeePredictor`**

**Fungsi**: Main class untuk training dan prediction dengan semua model

**Cara Kerja**:

1. **`__init__()`**
   - Membuat instance 4 model dengan parameter default:
     - `RandomForestRegressor`: 100 trees
     - `XGBoost`: 100 estimators
     - `CatBoost`: 100 estimators
     - `LightGBM`: 100 estimators
   - Semua dibungkus dengan `MultiOutputRegressor` untuk prediksi simultan 11 output (UKT-1 hingga UKT-11)

2. **`fit_models(X_train, y_train)`**
   - Melatih keempat model pada training data
   - Untuk setiap model, juga membuat quantile models
   - Menyimpan model yang sudah dilatih ke `trained_models` dan `quantile_models`

3. **`predict(X_test)`**
   - Memprediksi untuk semua model yang sudah dilatih
   - Mengembalikan dictionary: `{'RandomForest': predictions, 'XGBoost': predictions, ...}`
   - Setiap predictions adalah array 2D: `(n_samples, 11)` untuk 11 level UKT

4. **`predict_quantiles(X_test)`**
   - Memprediksi quantiles (0.1, 0.5, 0.9) untuk semua model
   - Mengembalikan dictionary dengan struktur:
     ```python
     {
         'RandomForest': {0.1: predictions, 0.5: predictions, 0.9: predictions},
         ...
     }
     ```

5. **`save_model()` / `load_model()`**
   - Menyimpan/memuat model menggunakan `joblib` (format pickle)

**Multi-Output Regression**:
- Setiap model memprediksi 11 nilai sekaligus (UKT-1 hingga UKT-11)
- `MultiOutputRegressor` melatih 11 regressor terpisah, satu untuk setiap output
- Lebih efisien daripada melatih 11 model terpisah karena sharing information

---

### 4. **src/conformal_prediction.py** - Conformal Prediction

**Fungsi**: Implementasi conformal prediction untuk uncertainty quantification yang tidak bergantung pada distribusi data

#### **Kelas: `ConformalPredictor`** (Single Output)

**Konsep Conformal Prediction**:
- Metode statistik untuk menghasilkan prediction intervals dengan jaminan coverage
- Tidak memerlukan asumsi distribusi tertentu
- Memberikan jaminan probabilistik: interval akan mengandung true value dengan probabilitas tertentu

**Cara Kerja**:

1. **`fit(X, y, ...)`**
   - Membagi data menjadi training dan calibration set
   - Melatih model base pada training set
   - Menghitung **non-conformity scores** pada calibration set:
     - `score = |y_true - y_pred|` (absolute residual)
   - Menyimpan scores ini untuk menghitung threshold

2. **`predict(X)`**
   - Memprediksi nilai dengan model base
   - Menghitung quantile dari calibration errors untuk significance level
     - Jika `significance_level = 0.1`, ambil quantile 90% dari errors
   - Membuat interval: `[prediction - threshold, prediction + threshold]`
   - Mengembalikan (predictions, lower_bounds, upper_bounds)

**Contoh**:
- Jika 90% quantile dari calibration errors = 500,000 IDR
- Interval untuk prediksi 5,000,000 IDR: [4,500,000, 5,500,000]

#### **Kelas: `MultiOutputConformalPredictor`** (Multi Output)

**Fungsi**: Extends conformal prediction untuk multi-output

**Inovasi Metodologi: Shared Calibration**:
- Menggunakan **calibration split yang sama** untuk semua output
- Memastikan fairness dan konsistensi di antara 11 output (UKT-1 hingga UKT-11)
- Tanpa shared calibration, setiap output mungkin menggunakan split berbeda, menyebabkan bias

**Cara Kerja**:

1. **`fit(X, y, ...)`**
   - Membuat satu calibration split untuk semua output
   - Untuk setiap output (UKT-1 hingga UKT-11):
     - Membuat single-output model dari base multi-output model
     - Membuat `ConformalPredictor` untuk output tersebut
     - Fit dengan calibration split yang sama
   - Menyimpan 11 conformal predictors dalam list

2. **`predict(X)`**
   - Untuk setiap conformal predictor, ambil prediksi dan intervals
   - Stack hasil menjadi array 2D: `(n_samples, 11 outputs)`
   - Mengembalikan (predictions, lower_bounds, upper_bounds)

**Fungsi Helper**:
- `create_conformal_predictors()`: Membuat conformal predictors untuk semua model
- `save_conformal_predictors()` / `load_conformal_predictors()`: Serialisasi menggunakan joblib

---

### 5. **src/evaluation.py** - Evaluasi Model

**Fungsi**: Menghitung metrik, membandingkan model, dan membuat visualisasi

#### **Kelas: `ModelEvaluator`**

**Metode Utama**:

1. **`calculate_metrics(y_true, y_pred, model_name)`**
   
   **Metrik yang Dihitung**:
   
   - **MSE (Mean Squared Error)**: `mean((y_true - y_pred)²)`
   - **RMSE (Root Mean Squared Error)**: `sqrt(MSE)`
   - **MAE (Mean Absolute Error)**: `mean(|y_true - y_pred|)`
   - **R² (Coefficient of Determination)**: 
     - `R² = 1 - (SS_res / SS_tot)`
     - Menunjukkan proporsi variance yang dijelaskan model
     - Range: -∞ hingga 1, semakin mendekati 1 semakin baik
   - **MAPE (Mean Absolute Percentage Error)**: 
     - `mean(|(y_true - y_pred) / y_true| * 100)`
     - Menangani zero values dengan safe calculation
   - **Pearson Correlation**: Korelasi linear antara prediksi dan actual
   
   **Handling Multi-Output**:
   - Menghitung metrik untuk setiap output secara terpisah
   - Menghitung metrik "Overall" pada semua output yang di-flatten
   - Mengembalikan dictionary dengan struktur nested

2. **`calculate_coverage(y_true, lower_bounds, upper_bounds)`**
   - Menghitung persentase actual values yang berada dalam prediction interval
   - Formula: `mean((y_true >= lower) & (y_true <= upper)) * 100`
   - Coverage seharusnya mendekati (1 - significance_level) * 100
   - Contoh: Untuk significance 0.1, coverage target adalah 90%

3. **`compare_models(results_dict)`**
   - Membandingkan beberapa model sekaligus
   - Input: Dictionary dengan format:
     ```python
     {
         'ModelName': {
             'predictions': y_pred,
             'y_true': y_true
         }
     }
     ```
   - Mengembalikan DataFrame dengan semua metrik untuk setiap model

4. **`plot_model_comparison(comparison_df)`**
   - Membuat bar chart untuk setiap metrik (RMSE, MAE, R², Pearson)
   - Menyimpan sebagai PNG files

5. **`create_interactive_comparison(comparison_df)`**
   - Membuat visualisasi interaktif menggunakan Plotly
   - Menyimpan sebagai HTML yang bisa dibuka di browser
   - Juga membuat versi matplotlib untuk PNG export

6. **`generate_report(comparison_df)`**
   - Membuat laporan Markdown lengkap
   - Memeringkat model berdasarkan R²
   - Menyertakan tabel perbandingan dan insights

**Metode Visualisasi Lainnya**:
- `plot_predictions()`: Scatter plot true vs predicted
- `plot_residuals()`: Residual plot untuk diagnosis model

---

### 6. **utils/visualization.py** - Utility Visualisasi

**Fungsi**: Fungsi-fungsi helper untuk visualisasi data dan hasil

**Fungsi Utama**:

1. **`visualize_data_distribution(data, target_cols)`**
   - Membuat histogram untuk setiap kolom UKT
   - Menunjukkan distribusi biaya kuliah di setiap level

2. **`visualize_feature_importance(models, feature_names)`**
   - Menghitung dan memvisualisasikan feature importance untuk setiap model
   - Untuk Random Forest: menggunakan `feature_importances_`
   - Untuk ensemble multi-output: rata-rata importance dari semua output
   - Menampilkan top 10 fitur paling penting

3. **`plot_prediction_intervals(y_true, y_pred, lower, upper, model_name)`**
   - Memvisualisasikan prediction intervals dari conformal prediction
   - Menampilkan true values, predictions, dan interval confidence
   - Membantu melihat seberapa baik interval menangkap actual values

4. **`plot_quantile_predictions(quantile_predictions, y_true, model_name)`**
   - Memvisualisasikan prediksi dari quantile regression
   - Menampilkan quantile 0.1 (lower), 0.5 (median), dan 0.9 (upper)
   - Membandingkan dengan actual values

5. **`correlation_heatmap(data, target_cols)`**
   - Membuat heatmap korelasi antar kolom UKT
   - Menunjukkan hubungan antara UKT level yang berbeda
   - Biasanya UKT level berurutan memiliki korelasi tinggi

6. **`analyze_prediction_accuracy(y_true, y_pred, threshold_percentage=0.1)`**
   - Menganalisis persentase prediksi yang berada dalam threshold tertentu
   - Contoh: Berapa persen prediksi yang berada dalam ±10% dari actual value
   - Mengembalikan dictionary dengan berbagai statistik akurasi

---

### 7. **src/main.py** - Main Execution Script

**Fungsi**: Script utama yang mengorchestrasi seluruh pipeline

**Alur Eksekusi Lengkap**:

```python
1. Data Preprocessing
   ├─ Load data dari CSV
   ├─ Visualize distribution
   ├─ Preprocess (encode, fill missing, etc.)
   ├─ Split train/test
   └─ Scale features

2. Model Training
   ├─ Initialize TuitionFeePredictor
   ├─ Train 4 models (RF, XGB, CatBoost, LightGBM)
   └─ Train quantile models untuk setiap base model

3. Model Prediction
   ├─ Predict dengan semua models
   └─ Predict quantiles dengan semua models

4. Conformal Prediction
   ├─ Create conformal predictors untuk semua models
   ├─ Fit conformal predictors dengan calibration split
   └─ Get prediction intervals

5. Model Evaluation
   ├─ Calculate metrics untuk semua models
   ├─ Calculate coverage untuk conformal intervals
   └─ Compare models

6. Visualization and Analysis
   ├─ Plot model comparison
   ├─ Visualize feature importance
   ├─ Plot prediction intervals
   ├─ Plot quantile predictions
   └─ Analyze prediction accuracy

7. Save Results
   ├─ Save comparison CSV
   ├─ Generate markdown report
   ├─ Save all trained models
   └─ Save conformal predictors
```

**Detail Setiap Step**:

**Step 1: Data Preprocessing**
- Load `Data/data.csv`
- Identifikasi kolom UKT (UKT-1 hingga UKT-11)
- Visualisasi distribusi dan correlation heatmap
- Preprocess dengan `DataPreprocessor`
- Split 80/20 untuk train/test
- Scale features dengan StandardScaler

**Step 2: Model Training**
- Buat instance `TuitionFeePredictor`
- Panggil `fit_models()` yang akan:
  - Train 4 base models untuk point predictions
  - Train 3 quantile models per base model (untuk q=0.1, 0.5, 0.9)
  - Total: 4 base + 12 quantile = 16 models

**Step 3: Model Prediction**
- `predict()` menghasilkan point predictions untuk semua models
- `predict_quantiles()` menghasilkan quantile predictions

**Step 4: Conformal Prediction**
- Untuk setiap trained model, buat `MultiOutputConformalPredictor`
- Fit dengan calibration split (20% dari training data)
- Predict dengan intervals untuk semua test samples

**Step 5: Model Evaluation**
- Hitung semua metrik (MSE, RMSE, MAE, R², MAPE, Pearson, Coverage)
- Bandingkan semua models dalam DataFrame
- Tentukan best model berdasarkan R²

**Step 6: Visualization**
- Plot comparison charts
- Visualisasi feature importance
- Plot prediction intervals (sample)
- Plot quantile predictions (sample)
- Analyze accuracy untuk best model

**Step 7: Save Results**
- Save `model_comparison.csv`
- Generate `research_report.md`
- Save semua models ke `models/` directory
- Save conformal predictors

---

### 8. **webapp/app.py** - Aplikasi Web Flask

**Fungsi**: Interface web untuk prediksi UKT secara interaktif

**Cara Kerja**:

1. **`load_models()`**
   - Load trained CatBoost model (best model) dari disk
   - Load conformal predictors jika tersedia
   - Load dan fit preprocessor untuk encoding dan scaling
   - Setup global variables untuk digunakan di routes

2. **Routes**:

   - **`/` (index)**
     - Menampilkan form input dengan dropdowns
     - Dropdown values diambil dari dataset (universitas, program, tahun, dll)
     - User mengisi form dan submit untuk prediksi
   
   - **`/api/predict` (POST)**
     - Menerima JSON dengan input features
     - Validate input fields
     - Encode categorical features menggunakan label encoders
     - Scale features menggunakan scaler
     - Predict dengan loaded model
     - Jika conformal predictors tersedia, hitung prediction intervals
     - Format response dengan prediksi untuk semua 11 UKT levels
     - Return JSON dengan format:
       ```json
       {
         "success": true,
         "predictions": {
           "UKT-1": {"value": 0, "formatted": "Rp 0", "lower": ..., "upper": ...},
           ...
         },
         "has_intervals": true
       }
       ```
   
   - **`/api/model-info` (GET)**
     - Mengembalikan informasi tentang model
     - Performance metrics, jumlah outputs, features, dll.
   
   - **`/api/health` (GET)**
     - Health check endpoint
     - Mengecek apakah models sudah di-load

3. **Error Handling**:
   - Jika model tidak ditemukan, return error page
   - Jika input tidak valid, return JSON error
   - Exception handling dengan traceback untuk debugging

**Frontend Integration**:
- Templates HTML di `webapp/templates/`
- Static files (CSS, JS) di `webapp/static/`
- JavaScript mengirim AJAX request ke `/api/predict`
- Menampilkan hasil prediksi dengan formatting yang baik

---

### 9. **tests/test_project.py** - Unit Tests

**Fungsi**: Test suite untuk memastikan semua komponen bekerja dengan benar

**Test Classes**:

1. **`TestDataPreprocessor`**
   - Test loading data
   - Test preprocessing
   - Test data splitting
   - Test feature scaling

2. **`TestTuitionFeePredictor`**
   - Test model training
   - Test predictions
   - Test quantile predictions

3. **`TestConformalPredictor`**
   - Test single-output conformal prediction
   - Test multi-output conformal prediction
   - Verify interval validity

4. **`TestModelEvaluator`**
   - Test metric calculations
   - Test coverage calculation
   - Test model comparison

**Run Tests**: `python -m pytest tests/test_project.py -v`

---

## Alur Data Lengkap

### Training Phase

```
CSV Data (Data/data.csv)
  ↓
DataPreprocessor.load_data()
  ↓
DataPreprocessor.preprocess()
  ├─ Separate features (Universitas, Program, etc.)
  ├─ Separate targets (UKT-1 hingga UKT-11)
  ├─ Label encoding untuk kategorikal
  └─ Fill missing values
  ↓
DataPreprocessor.split_data()
  ├─ X_train, y_train (80%)
  └─ X_test, y_test (20%)
  ↓
DataPreprocessor.scale_features()
  ├─ Fit StandardScaler pada X_train
  └─ Transform X_train dan X_test
  ↓
TuitionFeePredictor.fit_models()
  ├─ Train RandomForest
  ├─ Train XGBoost
  ├─ Train CatBoost
  ├─ Train LightGBM
  └─ Train quantile models untuk setiap base model
  ↓
create_conformal_predictors()
  ├─ Create MultiOutputConformalPredictor untuk setiap model
  └─ Fit dengan calibration split
  ↓
Save models ke disk (models/*.pkl)
```

### Prediction Phase

```
New Input Data
  ↓
Encode categorical features (menggunakan label encoders)
  ↓
Scale features (menggunakan fitted scaler)
  ↓
Model.predict()
  ├─ RandomForest → 11 predictions (UKT-1 to UKT-11)
  ├─ XGBoost → 11 predictions
  ├─ CatBoost → 11 predictions
  └─ LightGBM → 11 predictions
  ↓
Conformal Predictor.predict()
  ├─ Point predictions
  └─ Prediction intervals (lower, upper) untuk setiap output
  ↓
Format results dengan currency formatting
```

---

## Konsep-Konsep Penting

### 1. Multi-Output Regression

**Definisi**: Memrediksi beberapa target variables sekaligus

**Mengapa Multi-Output?**
- UKT-1 hingga UKT-11 saling berkorelasi
- Training bersama memungkinkan model belajar hubungan antar level
- Lebih efisien daripada melatih 11 model terpisah

**Implementasi**:
- Menggunakan `MultiOutputRegressor` dari scikit-learn
- Membungkus single-output estimator
- Melatih 11 regressor secara paralel, satu per output

### 2. Quantile Regression

**Definisi**: Memrediksi quantile dari distribusi target, bukan hanya mean

**Quantiles yang Digunakan**:
- **0.1 (10th percentile)**: Lower bound, 10% data di bawah ini
- **0.5 (50th percentile / median)**: Central estimate
- **0.9 (90th percentile)**: Upper bound, 90% data di bawah ini

**Manfaat**:
- Memberikan interval prediksi yang tidak simetris
- Berguna ketika error distribution tidak normal
- Memberikan informasi tentang uncertainty

**Implementasi**:
- XGBoost, CatBoost, LightGBM memiliki native quantile loss functions
- Random Forest tidak memiliki native support, menggunakan approximation

### 3. Conformal Prediction

**Definisi**: Framework untuk menghasilkan prediction sets dengan jaminan coverage

**Prinsip**:
1. Split data menjadi training dan calibration sets
2. Latih model pada training set
3. Hitung non-conformity scores pada calibration set
4. Gunakan quantile dari scores sebagai threshold untuk intervals

**Jaminan**: Coverage akan mendekati (1 - α) × 100% di mana α adalah significance level

**Shared Calibration** (Inovasi Proyek Ini):
- Menggunakan calibration split yang sama untuk semua 11 outputs
- Memastikan konsistensi dan fairness
- Mencegah bias yang mungkin muncul dari split berbeda

### 4. Feature Scaling

**Mengapa Penting?**
- Beberapa algoritma (XGBoost, LightGBM) kurang sensitif, tapi tetap membantu
- StandardScaler mengubah features ke mean=0, std=1
- Membantu konvergen lebih cepat dan stabil

**Formula**:
```
z = (x - μ) / σ
```

### 5. Label Encoding

**Mengapa Diperlukan?**
- Machine learning models hanya bisa bekerja dengan numerik
- Kategorikal seperti "UGM", "ITB" perlu dikonversi ke angka

**Implementasi**:
- Menggunakan `LabelEncoder` dari scikit-learn
- "UGM" → 0, "ITB" → 1, "UI" → 2, dll.
- Mapping disimpan untuk inverse transform saat diperlukan

---

## Metrik Evaluasi

### 1. **RMSE (Root Mean Squared Error)**
- **Formula**: `√(mean((y_true - y_pred)²))`
- **Interpretasi**: Average error dalam unit yang sama dengan target (IDR)
- **Sifat**: Memberikan penalti lebih besar untuk error besar

### 2. **MAE (Mean Absolute Error)**
- **Formula**: `mean(|y_true - y_pred|)`
- **Interpretasi**: Average absolute error
- **Sifat**: Lebih robust terhadap outliers daripada RMSE

### 3. **R² (Coefficient of Determination)**
- **Formula**: `1 - (SS_res / SS_tot)`
- **Range**: -∞ hingga 1
- **Interpretasi**: 
  - R² = 1: Perfect fit
  - R² = 0: Model tidak lebih baik daripada mean
  - R² < 0: Model lebih buruk daripada mean
- **Best Model**: CatBoost dengan R² = 0.9620 (96.2% variance explained)

### 4. **MAPE (Mean Absolute Percentage Error)**
- **Formula**: `mean(|(y_true - y_pred) / y_true| * 100)`
- **Interpretasi**: Average error dalam persentase
- **Keuntungan**: Mudah diinterpretasikan (misalnya "error rata-rata 7.96%")

### 5. **Pearson Correlation**
- **Range**: -1 hingga 1
- **Interpretasi**: Kekuatan hubungan linear antara prediksi dan actual
- **1**: Perfect positive correlation
- **0**: Tidak ada korelasi linear

### 6. **Coverage**
- **Formula**: `mean((y_true >= lower) & (y_true <= upper)) * 100`
- **Interpretasi**: Persentase actual values yang berada dalam prediction interval
- **Target**: Mendekati (1 - significance_level) × 100%
- **Best Model**: RandomForest dengan 91.10% coverage (target 90%)

---

## Hasil Penelitian

### Performance Ranking

| Rank | Model | R² | RMSE | MAE | MAPE | Coverage |
|------|-------|-----|------|-----|------|----------|
| 1 | **CatBoost** | **0.9620** | **1,102,068** | **449,465** | **7.96%** | 88.91% |
| 2 | LightGBM | 0.9574 | 1,166,708 | 519,485 | 8.95% | 89.30% |
| 3 | RandomForest | 0.9514 | 1,246,322 | 448,081 | 7.55% | **91.10%** |
| 4 | XGBoost | 0.9417 | 1,365,083 | 489,653 | 8.58% | 89.75% |

### Key Findings

1. **CatBoost**: Best overall performance dengan R² tertinggi (96.2%)
2. **RandomForest**: Best coverage (91.10%), melebihi target 90%
3. **Semua Model**: Excellent performance dengan R² > 0.94
4. **Practical Accuracy**: 77.76% prediksi CatBoost berada dalam ±10% dari actual value

---

## Cara Menggunakan Proyek

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Main Analysis

```bash
# Windows
run_research.bat

# Linux/Mac
python src/main.py
```

Ini akan:
- Preprocess data
- Train semua models
- Evaluate dan compare
- Generate visualizations
- Save results

### 3. Run Web Application

```bash
cd webapp
python app.py
```

Buka browser ke `http://localhost:5000`

### 4. Run Tests

```bash
python -m pytest tests/test_project.py -v
```

---

## Kesimpulan

Proyek ini adalah sistem prediksi UKT yang komprehensif dengan:

1. **4 Model Machine Learning**: Random Forest, XGBoost, CatBoost, LightGBM
2. **Multi-Output Regression**: Prediksi 11 level UKT sekaligus
3. **Quantile Regression**: Uncertainty quantification dengan 3 quantiles
4. **Conformal Prediction**: Prediction intervals dengan jaminan coverage
5. **Comprehensive Evaluation**: Multiple metrics untuk comparison
6. **Web Interface**: Flask app untuk prediksi interaktif
7. **Thorough Testing**: Unit tests untuk semua komponen

**Teknologi yang Digunakan**:
- Python 3.8+
- scikit-learn: Preprocessing, evaluation, base models
- XGBoost, CatBoost, LightGBM: Gradient boosting algorithms
- pandas, numpy: Data manipulation
- matplotlib, seaborn, plotly: Visualization
- Flask: Web framework
- joblib: Model serialization

**Inovasi Metodologi**:
- Shared calibration untuk multi-output conformal prediction
- Comprehensive comparison protocol
- Production-ready web interface

Proyek ini siap digunakan untuk prediksi UKT dan dapat di-extend untuk use cases lain yang memerlukan multi-output regression dengan uncertainty quantification.

