"""
Solar PV Fault Detection Pipeline
================================
End-to-end, production-minded ML workflow for classifying normal vs. faulty
conditions in photovoltaic systems, with optional fault-type classification.

Data schema expected (minimum):
    - timestamp
    - voltage
    - current
    - power_output
    - irradiance
    - temperature

Optional labels:
    - label_binary: 0 (normal) / 1 (fault)
    - fault_type: one of [degradation, shading, inverter_failure,
      wiring_issue, hotspot, normal]

Usage example:
    python pv_fault_detection_pipeline.py \
        --train_csv train.csv \
        --target label_binary \
        --timestamp_col timestamp \
        --mode binary

    # multiclass (if fault_type is available)
    python pv_fault_detection_pipeline.py \
        --train_csv train.csv \
        --target fault_type \
        --timestamp_col timestamp \
        --mode multiclass
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Configuration
# ----------------------------

PV_CORE_COLUMNS = [
    "voltage",
    "current",
    "power_output",
    "irradiance",
    "temperature",
]


@dataclass
class TrainArtifacts:
    model: Pipeline
    features: List[str]
    threshold: float
    metrics: Dict[str, float]


# ----------------------------
# Feature Engineering
# ----------------------------

def add_pv_domain_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Feature engineering ideas specific to PV systems:
      1) Theoretical power vs measured power gap
      2) Performance ratio-like metrics
      3) Temperature-corrected efficiency proxies
      4) Rolling statistics and rate-of-change (time-series aware)
      5) Time-of-day and seasonal context

    Notes:
      - Keep operations causal for real-time deployment by using trailing windows.
      - Replace impossible values and infs with NaN for robust imputation.
    """
    out = df.copy()

    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out = out.sort_values(timestamp_col).reset_index(drop=True)

    # Core sanitization
    for c in PV_CORE_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Domain features
    eps = 1e-6
    out["calc_power"] = out["voltage"] * out["current"]
    out["power_gap"] = out["power_output"] - out["calc_power"]
    out["power_gap_abs"] = out["power_gap"].abs()

    # Irradiance-normalized output: useful for shading/degradation detection
    out["power_per_irradiance"] = out["power_output"] / (out["irradiance"] + eps)

    # Temperature interaction (cell temp impacts output)
    out["temp_power_interaction"] = out["temperature"] * out["power_output"]
    out["temp_irradiance_interaction"] = out["temperature"] * out["irradiance"]

    # Temporal features
    out["hour"] = out[timestamp_col].dt.hour
    out["day_of_week"] = out[timestamp_col].dt.dayofweek
    out["month"] = out[timestamp_col].dt.month

    # Cyclical encoding for hour/month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    # Time-series trailing windows (causal)
    windows = [3, 6, 12]  # adjust to sensor frequency
    for col in ["power_output", "voltage", "current", "irradiance", "temperature"]:
        for w in windows:
            out[f"{col}_roll_mean_{w}"] = out[col].rolling(w, min_periods=1).mean()
            out[f"{col}_roll_std_{w}"] = out[col].rolling(w, min_periods=1).std()
        out[f"{col}_diff_1"] = out[col].diff(1)
        out[f"{col}_pct_change_1"] = out[col].pct_change(1)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


# ----------------------------
# Modeling
# ----------------------------

def _build_estimator(mode: str, random_state: int = 42) -> Tuple[Pipeline, Dict[str, List]]:
    """
    Uses Random Forest baseline for robustness and fast inference.
    For near-real-time systems, tree ensembles often provide a good
    accuracy/latency trade-off.

    Class imbalance handling is via class_weight and threshold tuning.
    """
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample" if mode == "binary" else "balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[("num", numeric_pipe, slice(0, None))],
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", classifier),
        ]
    )

    param_space = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [None, 8, 12, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features": ["sqrt", "log2", None],
    }
    return pipe, param_space


def _optimize_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """
    Finds the threshold that maximizes precision while satisfying minimum recall.
    This reduces false negatives, critical for fault detection.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    candidate_idxs = np.where(recalls[:-1] >= min_recall)[0]
    if len(candidate_idxs) == 0:
        # fallback: maximize recall-weighted F-score
        f2_scores = (5 * precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-9)
        best_idx = int(np.nanargmax(f2_scores))
    else:
        best_idx = int(candidate_idxs[np.nanargmax(precisions[candidate_idxs])])

    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
    return best_threshold, metrics


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    mode: str = "binary",
    random_state: int = 42,
) -> TrainArtifacts:
    """
    Step-by-step approach:
      1) Sort by time and engineer causal features
      2) Chronological split to avoid leakage
      3) Hyperparameter tuning with TimeSeriesSplit
      4) Threshold optimization to prioritize recall in binary mode
      5) Report metrics and confusion matrix
    """
    fe = add_pv_domain_features(df, timestamp_col=timestamp_col)

    if target_col not in fe.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Keep only numeric predictors and remove target/timestamp
    drop_cols = [target_col, timestamp_col]
    X = fe.drop(columns=[c for c in drop_cols if c in fe.columns])
    X = X.select_dtypes(include=[np.number])

    y = fe[target_col]
    if mode == "binary":
        y = y.astype(int)

    # Chronological holdout (last 20% for validation)
    split_idx = int(len(fe) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    estimator, param_space = _build_estimator(mode=mode, random_state=random_state)

    cv = TimeSeriesSplit(n_splits=4)
    scoring = "average_precision" if mode == "binary" else "f1_weighted"

    tuner = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_space,
        n_iter=20,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    tuner.fit(X_train, y_train)

    best_model = tuner.best_estimator_

    metrics: Dict[str, float] = {}
    threshold = 0.5

    if mode == "binary":
        val_prob = best_model.predict_proba(X_val)[:, 1]
        threshold, metrics = _optimize_threshold_for_recall(y_val.to_numpy(), val_prob, min_recall=0.95)
        val_pred = (val_prob >= threshold).astype(int)

        print("\nBinary validation report:")
        print(classification_report(y_val, val_pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_val, val_pred))
        print(f"Chosen threshold: {threshold:.4f}")
        print("Metrics:", json.dumps(metrics, indent=2))
    else:
        val_pred = best_model.predict(X_val)
        report = classification_report(y_val, val_pred, output_dict=True, zero_division=0)
        metrics = {
            "f1_weighted": float(report["weighted avg"]["f1-score"]),
            "precision_weighted": float(report["weighted avg"]["precision"]),
            "recall_weighted": float(report["weighted avg"]["recall"]),
        }
        print("\nMulticlass validation report:")
        print(classification_report(y_val, val_pred, digits=4, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_val, val_pred))
        print("Metrics:", json.dumps(metrics, indent=2))

    return TrainArtifacts(
        model=best_model,
        features=X.columns.tolist(),
        threshold=threshold,
        metrics=metrics,
    )


# ----------------------------
# Deployment Helpers
# ----------------------------

def save_artifacts(artifacts: TrainArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, output_dir / "pv_fault_model.joblib")
    meta = {
        "features": artifacts.features,
        "threshold": artifacts.threshold,
        "metrics": artifacts.metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def score_realtime_batch(
    incoming_df: pd.DataFrame,
    model: Pipeline,
    feature_order: List[str],
    timestamp_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Near-real-time scoring for small batches.
    Returns fault probabilities and binary decisions.
    """
    fe = add_pv_domain_features(incoming_df, timestamp_col=timestamp_col)
    X = fe.select_dtypes(include=[np.number]).reindex(columns=feature_order)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        fe["fault_probability"] = prob
        fe["fault_pred"] = pred
    else:
        fe["fault_pred"] = model.predict(X)

    return fe


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PV fault detection model")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--target", type=str, required=True, help="Target column")
    parser.add_argument("--timestamp_col", type=str, default="timestamp", help="Timestamp column")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.train_csv)

    artifacts = train_and_evaluate(
        df=df,
        target_col=args.target,
        timestamp_col=args.timestamp_col,
        mode=args.mode,
    )
    save_artifacts(artifacts, Path(args.output_dir))
    print(f"Saved model artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
