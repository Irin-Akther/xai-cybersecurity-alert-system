"""
Training script for the XAI Cybersecurity Alert System.

Usage:
    python train.py                              # auto-detect data/cicids2017.csv or synthetic
    python train.py --data data/your.csv        # train on a specific CICIDS2017 CSV
    python train.py --estimators 200            # custom Random Forest parameters
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.threat_detector import (
    ThreatDetector, FEATURE_COLS, LABEL_COL,
    _load_csv, _preprocess, _generate_synthetic_data,
)


# Preferred filenames checked in order before scanning the whole data/ directory
_PREFERRED_FILES = ["cicids2017.csv", "CICIDS2017.csv"]


def _find_data(explicit_path: Path | None) -> Path | None:
    """Return the CSV path to use, or None if synthetic data should be used."""
    if explicit_path is not None:
        return explicit_path
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        return None
    for name in _PREFERRED_FILES:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    # Fall back to any CSV in data/
    csvs = sorted(data_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def _print_dataset_summary(df, data_source: str):
    """Print class distribution and dataset statistics."""
    import pandas as pd
    import numpy as np

    total_rows = len(df)
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    label_col = None
    for col in df.columns:
        if col.strip().lower() == "label":
            label_col = col
            break

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Source      : {data_source}")
    print(f"  Total rows  : {total_rows:,}")
    print(f"  Features    : {len(available_features)} / {len(FEATURE_COLS)} CICIDS2017 columns")

    if label_col:
        labels = df[label_col].str.strip()
        benign = (labels == "BENIGN").sum()
        attacks = (labels != "BENIGN").sum()
        print(f"  BENIGN rows : {benign:,}  ({benign/total_rows:.1%})")
        print(f"  ATTACK rows : {attacks:,}  ({attacks/total_rows:.1%})")
        print("\n  Attack-type breakdown:")
        attack_counts = labels[labels != "BENIGN"].value_counts()
        for atype, count in attack_counts.items():
            print(f"    {atype:<40} {count:>8,}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XAI Threat Detection model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to a CICIDS2017 CSV (auto-detects data/cicids2017.csv if omitted).",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum tree depth (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory to save the trained model (default: models/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Resolve data source ---
    csv_path = _find_data(args.data)
    if args.data and not args.data.exists():
        logger.error("Data file not found: %s", args.data)
        sys.exit(1)

    # --- Load data for summary printout ---
    if csv_path and csv_path.exists():
        import pandas as pd
        logger.info("Loading dataset: %s", csv_path)
        df_raw = pd.read_csv(csv_path, low_memory=False)
        df_raw.columns = df_raw.columns.str.strip()
        _print_dataset_summary(df_raw, str(csv_path))
        data_dir = csv_path.parent
    else:
        logger.warning("No CICIDS2017 CSV found — using synthetic data (5,000 rows).")
        logger.info("  To use real data: download from unb.ca/cic/datasets/ids-2017.html")
        logger.info("  and place it at: %s", PROJECT_ROOT / "data" / "cicids2017.csv")
        df_raw = _generate_synthetic_data()
        _print_dataset_summary(df_raw, "synthetic (CICIDS2017-schema)")
        data_dir = PROJECT_ROOT / "data"

    # --- Train ---
    logger.info("Initialising ThreatDetector (n_estimators=%d, max_depth=%d)",
                args.estimators, args.max_depth)
    detector = ThreatDetector(
        n_estimators=args.estimators,
        max_depth=args.max_depth,
    )

    logger.info("Starting training…")
    metrics = detector.train(data_dir=data_dir)

    # --- Print training results ---
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"  Accuracy : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    report = metrics["report"]
    for label in ["BENIGN", "ATTACK"]:
        if label in report:
            r = report[label]
            print(
                f"  {label:<8}  precision={r['precision']:.3f}  "
                f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}  "
                f"support={int(r['support']):,}"
            )
    print("=" * 60 + "\n")

    # --- Save model ---
    logger.info("Saving model to %s", args.output)
    detector.save(args.output)

    metrics_path = args.output / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "accuracy": metrics["accuracy"],
        "report": {k: v for k, v in metrics["report"].items() if isinstance(v, dict)},
    }
    metrics_path.write_text(json.dumps(serialisable, indent=2))
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
