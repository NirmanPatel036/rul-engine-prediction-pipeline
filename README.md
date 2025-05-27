# Hybrid Predictive Maintenance Pipeline | NASA C-MAPSS Dataset

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive machine learning pipeline for **Remaining Useful Life (RUL)** prediction and **risk assessment** of aircraft turbofan engines using NASA's C-MAPSS dataset. This project implements a hybrid approach combining classification and regression models to enable proactive predictive maintenance.

## 🎯 Project Overview

This end-to-end pipeline processes multivariate time series data from aircraft engines to predict degradation stages and estimate remaining useful life, enabling optimal maintenance scheduling and reducing operational costs.

### Key Features
- **Multi-phase Pipeline**: Data preprocessing, classification, regression, and risk scoring
- **Advanced Feature Engineering**: 85 engineered features including smoothing, derivatives, and moving statistics
- **Hybrid ML Approach**: Combined classification (degradation stages) and regression (time-to-failure)
- **Intelligent Risk Scoring**: Prioritized maintenance recommendations based on failure probability
- **High Performance**: 97% classification accuracy with 98.9% F1-score on critical stages

## 📊 Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
- **20,631** multivariate time series samples
- **158** different engines
- **24** sensor measurements + 3 operational settings
- Multiple degradation patterns and operating conditions

## 🏗️ Pipeline Architecture

### Phase 1: Data Preprocessing & Feature Engineering
```python
# Key Operations:
- Sensor variance analysis (removed 6 low-variance sensors)
- RUL calculation and lifecycle analysis
- Advanced feature engineering (Savitzky-Golay filtering)
- K-means clustering for degradation stage identification
- Correlation analysis and feature selection
```

### Phase 2: Classification Model
```python
# Degradation Stage Classification:
- 5 stages: Normal → Slightly Degraded → Moderately Degraded → Critical → Failure
- SMOTE balancing (68 → 7,724 samples per class)
- XGBoost with hyperparameter optimization
- 97% accuracy, 98% F1-score
```

### Phase 3: Regression Model
```python
# Time-to-Next-Stage Prediction:
- Multiple regression models (Random Forest, XGBoost, Ridge)
- Target scaling and advanced preprocessing
- Hyperparameter tuning with RandomizedSearchCV
- R² = 0.159 for optimal model
```

### Phase 4: Risk Scoring & Maintenance Recommendations
```python
# Intelligent Risk Assessment:
- Combined failure probability and time predictions
- Fleet-wide risk analysis (8,265 samples)
- 82.9% precision, 100% recall for failure detection
- Automated maintenance scheduling recommendations
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Libraries
```python
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
imbalanced-learn>=0.8.0
```

### Usage
```bash
# Run the complete pipeline
python main.py

# Or run individual phases
python phase1_preprocessing.py
python phase2_classification.py
python phase3_regression.py
python phase4_risk_scoring.py
```

## 📈 Results & Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Random Forest | 95% | 95% | 91% | 97% |
| XGBoost | **97%** | **98%** | **94%** | **97%** |
| Optimized XGBoost | **97%** | **98.9%** | **94%** | **98%** |

### Risk Assessment Performance
- **Precision**: 82.9% for failure prediction
- **Recall**: 100% for critical failures
- **Fleet Optimization**: 99.8% engines in normal operation
- **Maintenance Efficiency**: 0.2% engines flagged for immediate attention

## 📁 Project Structure
```
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── phase1_preprocessing.py
├── phase2_classification.py
├── phase3_regression.py
├── phase4_risk_scoring.py
├── saved_models/
│   ├── classification_model.pkl
│   ├── regression_model.pkl
│   └── scalers/
├── phase1_plots/
├── phase2_plots/
├── phase3_plots/
├── phase4_plots/
├── requirements.txt
└── README.md
```

## 🔧 Technical Highlights

### Advanced Feature Engineering
- **Savitzky-Golay Filtering**: Noise reduction and trend extraction
- **Rate of Change Analysis**: First and second derivatives
- **Moving Statistics**: Windows of 5, 10, and 20 cycles
- **Correlation-based Selection**: Top 8 sensors identified

### Machine Learning Techniques
- **SMOTE**: Synthetic minority oversampling for class balance
- **RandomizedSearchCV**: Efficient hyperparameter optimization  
- **Ensemble Methods**: Random Forest and XGBoost
- **PCA Visualization**: Dimensionality reduction for cluster analysis

### Model Optimization
- **Cross-validation**: 3-fold stratified CV
- **Grid Search**: 10 iterations for optimal parameters
- **Feature Scaling**: StandardScaler and MinMaxScaler
- **Target Scaling**: Improved regression convergence

## 📊 Visualizations

The pipeline generates comprehensive visualizations:
- RUL distribution and sensor correlation heatmaps
- Degradation stage progression over engine lifecycle
- Risk score evolution and maintenance recommendations
- Model performance metrics and confusion matrices

## 🏆 Business Impact

- **Proactive Maintenance**: Early identification of critical engines
- **Cost Reduction**: Optimized maintenance scheduling
- **Safety Enhancement**: 100% recall for failure detection
- **Operational Efficiency**: Reduced false alarms by 99.8%

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA**: For providing the C-MAPSS dataset
- **Kaggle Community**: For dataset accessibility and inspiration
- **Research Team**: Nirman Patel, Poojan Patel, Manjari Pandey, Kushpreet Singh

⭐ **Star this repository if you found it helpful!** ⭐
