"""
Evaluation script for the XAI Cybersecurity Alert System.

Produces accuracy, precision, recall, and F1-score metrics suitable for
patent specifications and research papers using a reproducible 20% holdout split.

Usage:
    python evaluate.py                          # auto-detect data source
    python evaluate.py --data data/cicids2017.csv
    python evaluate.py --model-dir models/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

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
        description="Evaluate the XAI Threat Detection model and generate patent-ready metrics."
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
        help="Directory containing the trained model (default: models/).",
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

    logger.warning("No CSV found; using synthetic data for evaluation.")
    return _generate_synthetic_data(), "synthetic"


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info("Loading model from %s", args.model_dir)
    detector = ThreatDetector.load(args.model_dir)

    # ------------------------------------------------------------------
    # Load & preprocess data (same pipeline as training)
    # ------------------------------------------------------------------
    df_raw, source = _load_data(args.data)
    logger.info("Data source: %s  (%d rows)", source, len(df_raw))

    X, y_binary = _preprocess(df_raw)

    # Keep original multi-class labels for per-attack breakdown
    original_labels = (
        df_raw[LABEL_COL].str.strip()
        if LABEL_COL in df_raw.columns
        else None
    )

    # Reproducible split — same random_state as training
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

    logger.info("Test rows: %d  (%.0f%% holdout)", len(X_test), TEST_SIZE * 100)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    y_pred = detector.predict(X_test)

    # ------------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("  CLASSIFICATION REPORT")
    print(sep)
    report_str = classification_report(
        y_test, y_pred,
        target_names=["BENIGN", "ATTACK"],
    )
    print(report_str)

    report_dict = classification_report(
        y_test, y_pred,
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
    )

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    print(f"{sep}")
    print("  CONFUSION MATRIX")
    print(sep)
    print(f"  {'':>28} {'Predicted BENIGN':>18} {'Predicted ATTACK':>18}")
    print(f"  {'Actual BENIGN':>28} {cm[0, 0]:>18,} {cm[0, 1]:>18,}")
    print(f"  {'Actual ATTACK':>28} {cm[1, 0]:>18,} {cm[1, 1]:>18,}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives (BENIGN correctly classified) : {tn:,}")
    print(f"  False Positives (BENIGN misclassified)        : {fp:,}")
    print(f"  False Negatives (ATTACK missed)               : {fn:,}")
    print(f"  True Positives  (ATTACK correctly detected)   : {tp:,}")

    # ------------------------------------------------------------------
    # Per-attack-type detection rate
    # ------------------------------------------------------------------
    attack_breakdown: dict[str, dict] = {}
    if labels_test is not None:
        print(f"\n{sep}")
        print("  PER-ATTACK-TYPE DETECTION RATE")
        print(sep)
        print(f"  {'Attack Type':<38} {'Total':>8} {'Detected':>10} {'Rate':>8}")
        print("  " + "-" * 66)

        attack_types = sorted(labels_test[labels_test != "BENIGN"].unique())
        for atype in attack_types:
            mask = labels_test == atype
            total = int(mask.sum())
            detected = int(y_pred[mask.values].sum())
            rate = detected / total if total > 0 else 0.0
            print(f"  {atype:<38} {total:>8,} {detected:>10,} {rate:>7.1%}")
            attack_breakdown[atype] = {
                "total": total,
                "detected": detected,
                "detection_rate": round(rate, 4),
            }

    # ------------------------------------------------------------------
    # Summary line (copy into patent specification)
    # ------------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    n_test = len(y_test)

    summary_line = (
        f"Overall Accuracy: {accuracy * 100:.1f}% | "
        f"Macro F1: {macro_f1 * 100:.1f}% | "
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
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "n_test_rows": n_test,
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "summary_line": summary_line,
        "classification_report": {
            k: v for k, v in report_dict.items() if isinstance(v, dict)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "per_attack_type": attack_breakdown,
    }
    out_path.write_text(json.dumps(metrics_out, indent=2))
    logger.info("Evaluation metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
