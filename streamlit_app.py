"""
streamlit_app.py

Streamlit dashboard for ML Model Monitoring System.

This dashboard:
- Displays alerts
- Shows model performance metrics
- Visualizes drift and degradation
- Contains NO monitoring logic
"""

import json
import os
import pandas as pd
import streamlit as st


# =========================
# 1. PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    layout="wide"
)

st.title("📊 ML Model Monitoring & Drift Detection Dashboard")


# =========================
# 2. FILE PATHS
# =========================

ALERTS_FILE = "reports/alerts/alerts_log.json"
BASELINE_METRICS_FILE = "reports/baseline/performance_metrics.json"
LATEST_METRICS_FILE = "reports/monitoring/latest_performance.json"
PERFORMANCE_HISTORY_FILE = "reports/monitoring/performance_history.json"
DATA_DRIFT_FILE = "reports/drift/data_drift_report.json"
CONCEPT_DRIFT_FILE = "reports/monitoring/concept_drift_report.json"


# =========================
# 3. HELPER FUNCTION
# =========================

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# =========================
# 4. ALERTS SECTION (TOP PRIORITY)
# =========================

st.header("🚨 Alerts")

alerts = load_json(ALERTS_FILE)

if alerts:
    alerts_df = pd.DataFrame(alerts).sort_values("timestamp", ascending=False)

    # Show most recent alerts
    st.dataframe(alerts_df.head(10), use_container_width=True)

    # Highlight critical alerts
    critical_count = (alerts_df["level"] == "CRITICAL").sum()
    warning_count = (alerts_df["level"] == "WARNING").sum()

    col1, col2 = st.columns(2)
    col1.metric("⚠️ Warnings", warning_count)
    col2.metric("🔥 Critical Alerts", critical_count)

else:
    st.success("✅ No alerts available.")


# =========================
# 5. PERFORMANCE METRICS
# =========================

st.header("📈 Model Performance")

baseline_metrics = load_json(BASELINE_METRICS_FILE)
latest_metrics = load_json(LATEST_METRICS_FILE)

if baseline_metrics and latest_metrics:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline (Training Time)")
        st.json(baseline_metrics)

    with col2:
        st.subheader("Latest (Production)")
        st.json(latest_metrics)
else:
    st.warning("Performance metrics not available.")


# =========================
# 6. PERFORMANCE TREND
# =========================

st.header("📉 Performance Trend Over Time")

performance_history = load_json(PERFORMANCE_HISTORY_FILE)

if performance_history:
    perf_df = pd.DataFrame(performance_history)
    perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])

    st.line_chart(
        perf_df.set_index("timestamp")[["accuracy", "precision", "recall"]]
    )
else:
    st.info("No performance history found.")


# =========================
# 7. DATA DRIFT SUMMARY
# =========================

st.header("🔍 Data Drift Summary")

data_drift = load_json(DATA_DRIFT_FILE)

if data_drift:
    drift_df = pd.DataFrame.from_dict(data_drift, orient="index")
    drifted = drift_df[drift_df["drift_detected"] == True]

    st.write(f"Drift detected in **{len(drifted)} features**.")

    if not drifted.empty:
        st.dataframe(drifted, use_container_width=True)
    else:
        st.success("✅ No data drift detected.")
else:
    st.warning("Data drift report not available.")


# =========================
# 8. CONCEPT DRIFT STATUS
# =========================

st.header("🧠 Concept Drift")

concept_drift = load_json(CONCEPT_DRIFT_FILE)

if concept_drift:
    if concept_drift.get("concept_drift_detected"):
        st.error("⚠️ Concept drift detected!")
    else:
        st.success("✅ No concept drift detected.")

    st.json(concept_drift)
else:
    st.warning("Concept drift report not available.")