# Solar PV Fault Detection Pipeline 

This repository contains a practical machine learning pipeline for **solar photovoltaic (PV) fault detection** from timestamped telemetry.

The implementation lives in: 
- `pv_fault_detection_pipeline.py`  

It supports: 
- **Binary classification**: normal vs fault 
- **Multiclass classification**: optional fault-type prediction
- **Time-series-aware training and validation**
- **Recall-prioritized thresholding** to reduce missed faults
- **Near-real-time batch scoring** helpers 

--- 

## 1) Problem Setup

Typical input columns:
- `timestamp`
- `voltage`
- `current`
- `power_output`
- `irradiance`
- `temperature`

Possible targets:
- `label_binary` (`0` normal, `1` fault)
- `fault_type` (e.g., degradation, shading, inverter failure, wiring issue, hotspot)

> Fault detection systems should prioritize **low false negatives** because missed faults can increase downtime and safety risk.

---

## 2) Modeling Approach (Step-by-step)

1. **Data cleaning + ordering**
   - Parse timestamps
   - Sort chronologically
   - Coerce sensor fields to numeric

2. **PV-specific feature engineering**
   - Calculated-vs-measured power discrepancy (`power_gap`)
   - Irradiance-normalized output (`power_per_irradiance`)
   - Temperature interaction features
   - Time-of-day / seasonality (cyclical encodings)
   - Causal rolling mean/std and short-term deltas

3. **Leakage-safe validation**
   - Chronological holdout split (last 20% as validation)
   - `TimeSeriesSplit` inside hyperparameter search

4. **Model training and tuning**
   - Random Forest baseline with class weights
   - `RandomizedSearchCV` for key tree parameters

5. **Decision threshold optimization (binary mode)**
   - Tune probability threshold to maximize precision while meeting a recall floor (default `min_recall=0.95`)

6. **Evaluation and reporting**
   - Binary: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
   - Multiclass: weighted precision/recall/F1 + confusion matrix

7. **Deployment artifacts**
   - Save trained model (`joblib`)
   - Save metadata (`features`, `threshold`, metrics)

---

## 3) Why this baseline

A tree-ensemble baseline is often a strong first choice in PV telemetry systems because it:
- works well on tabular sensor data,
- tolerates moderate noise and missingness after imputation,
- provides fast inference suitable for near-real-time alerting,
- can handle non-linear feature interactions without extensive manual transforms.

---

## 4) Handling class imbalance

The pipeline currently addresses rarity of faults via:
- class-weighted training (`balanced` / `balanced_subsample`),
- threshold tuning for high recall,
- PR-focused scoring (`average_precision`) in binary mode.

Additional options you can add:
- time-aware under/over-sampling on training folds only,
- cost-sensitive objectives,
- anomaly-first stage for candidate fault windows.

---

## 5) How to run

### Binary classification

```bash
python pv_fault_detection_pipeline.py \
  --train_csv train.csv \
  --target label_binary \
  --timestamp_col timestamp \
  --mode binary \
  --output_dir artifacts
```

### Multiclass classification

```bash
python pv_fault_detection_pipeline.py \
  --train_csv train.csv \
  --target fault_type \
  --timestamp_col timestamp \
  --mode multiclass \
  --output_dir artifacts
```

---

## 6) Real-time / near-real-time deployment guidance

For production monitoring systems:
- ingest telemetry in micro-batches,
- compute the same causal features online,
- score with `score_realtime_batch`,
- alert when `fault_probability >= threshold`,
- log predictions + outcomes for drift and retraining.

Recommended operational controls:
- retraining cadence (e.g., weekly/monthly),
- separate thresholds by site or climate zone,
- alert suppression/debouncing to reduce noisy repeats,
- model and feature drift dashboards.

---

## 7) Next improvements

- Add XGBoost/LightGBM variant for stronger tabular performance.
- Add sequence model (LSTM/Temporal CNN/Transformer) when high-frequency history is critical.
- Add explicit anomaly detector (Isolation Forest/Autoencoder) for unseen fault modes.
- Add unit tests and data contracts for schema validation.

