"""
Evaluation script for the XAI Cybersecurity Alert System.

Produces accuracy, precision, recall, F1-score, AUC, and per-attack detection
rates for three classifiers:
  1. Random Forest      — the proposed model
  2. Decision Tree      — interpretable single-tree baseline
  3. Logistic Regression — linear baseline (with StandardScaler pipeline)

All three use an identical 20% stratified holdout (random_state=42) for a
fair, reproducible comparison suitable for patent specifications and papers.

Usage:
    python evaluate.py                          # auto-detect data source
    python evaluate.py --data data/cicids2017.csv
    python evaluate.py --model-dir models/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.threat_detector import (
    FEATURE_COLS,
    LABEL_COL,
    ThreatDetector,
    _generate_synthetic_data,
    _load_csv,
    _preprocess,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the XAI Threat Detection model against two baselines "
            "and generate patent-ready metrics."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to a CICIDS2017 CSV (auto-detected if omitted).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory containing the trained Random Forest model (default: models/).",
    )
    return parser.parse_args()


def _load_data(explicit_path: Path | None) -> tuple[pd.DataFrame, str]:
    """Return (raw_df, source_description)."""
    if explicit_path is not None:
        if not explicit_path.exists():
            logger.error("Data file not found: %s", explicit_path)
            sys.exit(1)
        df = pd.read_csv(explicit_path, low_memory=False)
        df.columns = df.columns.str.strip()
        return df, str(explicit_path)

    data_dir = PROJECT_ROOT / "data"
    df = _load_csv(data_dir) if data_dir.exists() else None
    if df is not None:
        return df, str(data_dir)

    logger.warning("No CSV found — using synthetic data for evaluation.")
    return _generate_synthetic_data(), "synthetic"


def _metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    fit_seconds: float,
) -> dict:
    """Assemble a metrics dictionary for one model."""
    report = classification_report(
        y_true, y_pred,
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    auc = (
        float(roc_auc_score(y_true, y_proba))
        if y_proba is not None
        else None
    )
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "macro_f1": round(report["macro avg"]["f1-score"], 6),
        "macro_precision": round(report["macro avg"]["precision"], 6),
        "macro_recall": round(report["macro avg"]["recall"], 6),
        "auc_roc": round(auc, 6) if auc is not None else None,
        "fit_seconds": round(fit_seconds, 2),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": {
            k: v for k, v in report.items() if isinstance(v, dict)
        },
    }


def _print_comparison(results: dict[str, dict]) -> str:
    """Print and return the comparison table string."""
    sep = "=" * 76
    header = (
        f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>8} {'F1 Macro':>9} {'AUC-ROC':>8} {'Time(s)':>8}"
    )
    print(f"\n{sep}")
    print("  CLASSIFIER COMPARISON TABLE")
    print(sep)
    print(header)
    print("  " + "-" * 72)
    rows = []
    for name, m in results.items():
        auc_str = f"{m['auc_roc']:.4f}" if m["auc_roc"] is not None else "  N/A  "
        row = (
            f"  {name:<22} {m['accuracy']:>9.4f} {m['macro_precision']:>10.4f} "
            f"{m['macro_recall']:>8.4f} {m['macro_f1']:>9.4f} "
            f"{auc_str:>8} {m['fit_seconds']:>8.2f}"
        )
        print(row)
        rows.append(row)
    print(f"{sep}")
    return "\n".join([header] + rows)


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load & preprocess data
    # ------------------------------------------------------------------
    df_raw, source = _load_data(args.data)
    logger.info("Data source: %s  (%d rows)", source, len(df_raw))

    X, y_binary = _preprocess(df_raw)

    original_labels = (
        df_raw[LABEL_COL].str.strip() if LABEL_COL in df_raw.columns else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_binary,
    )

    if original_labels is not None:
        _, _, _, labels_test = train_test_split(
            X, original_labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_binary,
        )
    else:
        labels_test = None

    logger.info(
        "Train rows: %d  |  Test rows: %d  (%.0f%% holdout)",
        len(X_train), len(X_test), TEST_SIZE * 100,
    )

    # ------------------------------------------------------------------
    # Model 1 — Random Forest (load pre-trained or train fresh)
    # ------------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("  LOADING / TRAINING MODELS")
    print(sep)

    t0 = time.perf_counter()
    try:
        detector = ThreatDetector.load(args.model_dir)
        rf_model = detector.model
        rf_time = time.perf_counter() - t0
        logger.info("Random Forest loaded from disk (%.2fs)", rf_time)
        # Re-fit on the training split so fit time is meaningful
        t0 = time.perf_counter()
        rf_model.fit(X_train, y_train)
        rf_time = time.perf_counter() - t0
    except FileNotFoundError:
        logger.info("No saved model found — training Random Forest from scratch.")
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        t0 = time.perf_counter()
        rf_model.fit(X_train, y_train)
        rf_time = time.perf_counter() - t0

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = _metrics_dict(y_test, rf_pred, rf_proba, rf_time)
    logger.info(
        "Random Forest — Acc: %.4f  F1: %.4f  AUC: %.4f",
        rf_metrics["accuracy"], rf_metrics["macro_f1"], rf_metrics["auc_roc"],
    )

    # ------------------------------------------------------------------
    # Model 2 — Decision Tree baseline
    # ------------------------------------------------------------------
    dt_model = DecisionTreeClassifier(
        max_depth=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    t0 = time.perf_counter()
    dt_model.fit(X_train, y_train)
    dt_time = time.perf_counter() - t0
    dt_pred = dt_model.predict(X_test)
    dt_proba = dt_model.predict_proba(X_test)[:, 1]
    dt_metrics = _metrics_dict(y_test, dt_pred, dt_proba, dt_time)
    logger.info(
        "Decision Tree  — Acc: %.4f  F1: %.4f  AUC: %.4f",
        dt_metrics["accuracy"], dt_metrics["macro_f1"], dt_metrics["auc_roc"],
    )

    # ------------------------------------------------------------------
    # Model 3 — Logistic Regression baseline (StandardScaler pipeline)
    # ------------------------------------------------------------------
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",   # saga scales better than lbfgs on large datasets
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])
    t0 = time.perf_counter()
    lr_pipeline.fit(X_train, y_train)
    lr_time = time.perf_counter() - t0
    lr_pred = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = _metrics_dict(y_test, lr_pred, lr_proba, lr_time)
    logger.info(
        "Logistic Regr. — Acc: %.4f  F1: %.4f  AUC: %.4f",
        lr_metrics["accuracy"], lr_metrics["macro_f1"], lr_metrics["auc_roc"],
    )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    all_results = {
        "Random Forest (proposed)": rf_metrics,
        "Decision Tree": dt_metrics,
        "Logistic Regression": lr_metrics,
    }
    _print_comparison(all_results)

    # ------------------------------------------------------------------
    # Individual classification reports
    # ------------------------------------------------------------------
    for label, model, pred in [
        ("Random Forest (proposed)", rf_model, rf_pred),
        ("Decision Tree", dt_model, dt_pred),
        ("Logistic Regression", lr_pipeline, lr_pred),
    ]:
        print(f"\n{sep}")
        print(f"  CLASSIFICATION REPORT — {label.upper()}")
        print(sep)
        print(
            classification_report(
                y_test, pred,
                target_names=["BENIGN", "ATTACK"],
                zero_division=0,
            )
        )

    # ------------------------------------------------------------------
    # Per-attack-type detection rate (Random Forest only — proposed model)
    # ------------------------------------------------------------------
    attack_breakdown: dict[str, dict] = {}
    if labels_test is not None:
        print(f"\n{sep}")
        print("  PER-ATTACK-TYPE DETECTION RATE — Random Forest (proposed)")
        print(sep)
        print(f"  {'Attack Type':<38} {'Total':>8} {'Detected':>10} {'Rate':>8}")
        print("  " + "-" * 66)

        attack_types = sorted(labels_test[labels_test != "BENIGN"].unique())
        for atype in attack_types:
            mask = labels_test == atype
            total = int(mask.sum())
            detected = int(rf_pred[mask.values].sum())
            rate = detected / total if total > 0 else 0.0
            print(f"  {atype:<38} {total:>8,} {detected:>10,} {rate:>7.1%}")
            attack_breakdown[atype] = {
                "total": total,
                "detected": detected,
                "detection_rate": round(rate, 4),
            }

    # ------------------------------------------------------------------
    # Patent specification summary line
    # ------------------------------------------------------------------
    rf_acc = rf_metrics["accuracy"]
    rf_f1 = rf_metrics["macro_f1"]
    rf_auc = rf_metrics["auc_roc"]
    n_test = len(y_test)

    summary_line = (
        f"Proposed Model (Random Forest) — "
        f"Accuracy: {rf_acc * 100:.1f}% | "
        f"Macro F1: {rf_f1 * 100:.1f}% | "
        f"AUC-ROC: {rf_auc:.4f} | "
        f"Test rows: {n_test:,}"
    )

    print(f"\n{sep}")
    print("  PATENT SPECIFICATION SUMMARY LINE")
    print(sep)
    print(f"  {summary_line}")
    print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------
    out_path = args.model_dir / "evaluation_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "n_test_rows": n_test,
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "data_source": source,
        "summary_line": summary_line,
        "models": {
            "random_forest_proposed": rf_metrics,
            "decision_tree_baseline": dt_metrics,
            "logistic_regression_baseline": lr_metrics,
        },
        "per_attack_type_random_forest": attack_breakdown,
    }
    out_path.write_text(json.dumps(metrics_out, indent=2))
    logger.info("Evaluation metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
