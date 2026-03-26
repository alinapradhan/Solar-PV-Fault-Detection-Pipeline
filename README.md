# Solar Energy / Solar Cell Repositories

A curated starter list for photovoltaics (PV), simulation, and optimization projects.

## 🔬 Simulation & Physics

- **[pvlib-python](https://github.com/pvlib/pvlib-python)**  
  Widely used open-source Python library for irradiance modeling, module/inverter performance, and end-to-end PV system simulation.

- **[NREL System Advisor Model (SAM)](https://github.com/NREL/SAM)**  
  Industry-standard techno-economic modeling tool for detailed energy yield and financial analysis across solar and other technologies.

- **[PySolar](https://github.com/pingswept/pysolar)**  
  Lightweight Python library focused on solar position and clear-sky radiation calculations.

## ⚡ Solar Panel Optimization / ML

- **[pvanalytics](https://github.com/pvlib/pvanalytics)**  
  Data-quality and performance analytics toolkit for PV operational datasets (cleaning, anomaly detection, and QA checks).

- **[PV_ICE](https://github.com/NREL/PV_ICE)**  
  NREL's open-source modeling framework for PV energy/material flow analysis and scenario-based lifecycle optimization.

- **[power-predictor](https://github.com/SverreNystad/power-predictor)**  
  Practical ML forecasting example repo (includes LSTM and XGBoost workflows) for PV power prediction from weather features.

## 🔎 Useful GitHub Search Prompts

Use these directly in GitHub search for more projects and datasets:

- `solar power forecasting LSTM`
- `solar forecasting xgboost`
- `photovoltaic fault detection machine learning`
- `solar predictive maintenance dataset`
- `electroluminescence solar cell defect dataset`

## Notes

- For production planning/operations, combine a **physics model** (pvlib/SAM) with a **data-driven forecaster** (LSTM/XGBoost).
- Prefer repositories with active maintenance, tests/CI, clear licenses, and reproducible notebooks.
