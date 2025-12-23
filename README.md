# ML Model Monitoring System

A production-ready machine learning model monitoring system that provides **data drift detection**, **concept drift detection**, **performance monitoring**, and **automated alerting** capabilities.

##  Features

- **Model Training** - Train and save baseline ML models with performance metrics
- **Data Drift Detection** - Detect feature distribution changes using Kolmogorov-Smirnov tests
- **Concept Drift Detection** - Monitor prediction probability distribution shifts
- **Performance Monitoring** - Track model accuracy, precision, recall, and ROC-AUC over time
- **Alert Engine** - Generate warnings and critical alerts based on drift and degradation
- **Retraining Controller** - Automated decisions for model retraining
- **FastAPI Backend** - RESTful API for accessing monitoring data
- **Streamlit Dashboard** - Interactive visualization of metrics and alerts

##  Project Structure

```
ml_monitoring_system/
├── src/
│   ├── __init__.py              # Package init
│   ├── config.py                # Centralized configuration
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI backend
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── alert_engine.py      # Alert generation logic
│   │   ├── concept_drift.py     # Concept drift detection
│   │   ├── data_drift.py        # Data drift detection
│   │   ├── performance_monitor.py # Performance tracking
│   │   └── retrain_controller.py  # Retraining decisions
│   └── training/
│       ├── __init__.py
│       ├── train_model.py       # Model training pipeline
│       └── evaluate_model.py    # Model evaluation
├── models/
│   └── baseline/                # Saved model artifacts
├── reports/
│   ├── alerts/                  # Alert logs
│   ├── baseline/                # Baseline metrics & stats
│   ├── drift/                   # Drift detection reports
│   ├── monitoring/              # Performance history
│   └── retraining/              # Retraining decisions
├── tests/
│   ├── conftest.py              # Test fixtures
│   ├── test_api.py              # API tests
│   └── test_monitoring.py       # Monitoring tests
├── streamlit_app.py             # Dashboard application
├── run_pipeline.py              # Pipeline orchestration
├── requirements.txt             # Dependencies
└── .gitignore                   # Git ignore rules
```

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml_monitoring_system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python run_pipeline.py

# With verbose output:
python run_pipeline.py --verbose

# Skip training (use existing model):
python run_pipeline.py --skip-training
```

This will execute:
1. Train baseline model
2. Evaluate on simulated production data
3. Detect data drift
4. Detect concept drift
5. Monitor performance
6. Generate alerts
7. Make retraining decisions

### 3. Start the Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`

### 4. Start the API Server

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

##  API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/alerts` | GET | Get all alerts |
| `/metrics/baseline` | GET | Baseline performance metrics |
| `/metrics/latest` | GET | Latest performance metrics |
| `/metrics/history` | GET | Performance history over time |
| `/drift/data` | GET | Data drift report |
| `/drift/concept` | GET | Concept drift report |
| `/retraining/status` | GET | Retraining decision status |

##  Dashboard Features

- **Alerts Panel** - View recent warnings and critical alerts
- **Performance Metrics** - Compare baseline vs production metrics
- **Performance Trends** - Line charts showing accuracy, precision, recall over time
- **Data Drift Summary** - Features with detected distribution changes
- **Concept Drift Status** - Prediction distribution shift detection

##  Running Tests

```bash
pytest tests/ -v
```

##  Configuration

Key thresholds can be adjusted in `src/config.py` or respective modules:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `P_VALUE_THRESHOLD` | `config.py` | 0.05 | Statistical significance for drift |
| `DEGRADATION_THRESHOLD` | `config.py` | 0.90 | % of baseline before degradation |
| `CRITICAL_ALERT_THRESHOLD` | `config.py` | 1 | Critical alerts before retraining |
| `LOOKBACK_HOURS` | `config.py` | 24 | Time window for alert analysis |


##  License

MIT License

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

