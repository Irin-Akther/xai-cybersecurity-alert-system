"""
SHAP Consistency Analysis — measures intra-class SHAP feature rank stability.

For each major attack class, samples 100 test instances and computes the
pairwise Spearman rank correlation (rho) of their absolute SHAP value vectors.
A high mean rho means the model consistently relies on the same features to
detect that attack type — strong evidence of mechanistic interpretability.

Usage:
    python shap_consistency.py
    python shap_consistency.py --data data/cicids2017.csv --samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shap_consistency")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.threat_detector import (
    LABEL_COL,
    ThreatDetector,
    _generate_synthetic_data,
    _load_csv,
    _preprocess,
)
from modules.xai_explainer import XAIExplainer

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Attack classes with sufficient samples for meaningful pairwise statistics
TARGET_CLASSES = [
    "DDoS",
    "PortScan",
    "DoS Hulk",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHAP consistency analysis — intra-class feature rank stability."
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Max instances per attack class (default: 100).",
    )
    return parser.parse_args()


def _load_data(explicit_path: Path | None) -> tuple[pd.DataFrame, str]:
    if explicit_path is not None:
        if not explicit_path.exists():
            logger.error("File not found: %s", explicit_path)
            sys.exit(1)
        df = pd.read_csv(explicit_path, low_memory=False)
        df.columns = df.columns.str.strip()
        return df, str(explicit_path)
    data_dir = PROJECT_ROOT / "data"
    df = _load_csv(data_dir) if data_dir.exists() else None
    if df is not None:
        return df, str(data_dir)
    logger.warning("No CSV found — using synthetic data.")
    return _generate_synthetic_data(), "synthetic"


def _pairwise_spearman(abs_shap: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Compute pairwise Spearman rho between all rows of abs_shap (n_samples, n_features).

    Returns (mean_rho, std_rho, full_rho_vector).
    """
    n = abs_shap.shape[0]
    if n < 2:
        return float("nan"), float("nan"), np.array([])

    # spearmanr on the transposed matrix: each sample is a "variable"
    # → returns (n_samples, n_samples) correlation matrix
    result = spearmanr(abs_shap.T)
    # scipy ≥1.9: result.statistic; older: result[0]
    corr_matrix = np.asarray(
        getattr(result, "statistic", result[0]), dtype=float
    )

    if corr_matrix.ndim == 0:
        # Only two samples → scalar returned
        return float(corr_matrix), float("nan"), np.array([float(corr_matrix)])

    # Extract upper triangle (exclude diagonal)
    idx_i, idx_j = np.triu_indices(n, k=1)
    rho_vals = corr_matrix[idx_i, idx_j]
    return float(np.nanmean(rho_vals)), float(np.nanstd(rho_vals)), rho_vals


def _top3_agreement(abs_shap: np.ndarray, feature_names: list[str]) -> float:
    """
    What fraction of all sample pairs share ≥ 2 of their top-3 features?
    More interpretable complement to Spearman rho.
    """
    n = abs_shap.shape[0]
    if n < 2:
        return float("nan")

    top3_per_sample = [
        set(np.argsort(row)[-3:]) for row in abs_shap
    ]
    agree_count = 0
    total = 0
    for i, j in combinations(range(n), 2):
        if len(top3_per_sample[i] & top3_per_sample[j]) >= 2:
            agree_count += 1
        total += 1
    return agree_count / total if total > 0 else float("nan")


def _consensus_top_features(
    abs_shap: np.ndarray, feature_names: list[str], top_n: int = 5
) -> list[str]:
    """Features with the highest mean |SHAP| across all samples in the class."""
    mean_abs = abs_shap.mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    return [feature_names[i] for i in top_idx]


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info("Loading model from %s", args.model_dir)
    detector = ThreatDetector.load(args.model_dir)
    explainer = XAIExplainer(detector, top_n=60)   # top_n=60 → all features
    feature_names = list(detector.feature_names)   # 60 names

    # ------------------------------------------------------------------
    # Load & split data — identical split to evaluate.py
    # ------------------------------------------------------------------
    df_raw, source = _load_data(args.data)
    logger.info("Data source: %s  (%d rows)", source, len(df_raw))

    X, y_binary = _preprocess(df_raw)
    original_labels = (
        df_raw[LABEL_COL].str.strip() if LABEL_COL in df_raw.columns else None
    )

    _, X_test, _, y_test = train_test_split(
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
        logger.warning("No Label column — cannot filter by attack class. Exiting.")
        sys.exit(1)

    logger.info("Test set: %d rows", len(X_test))

    # Reset indices so loc/iloc are consistent
    X_test = X_test.reset_index(drop=True)
    labels_test = labels_test.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Per-class SHAP consistency analysis
    # ------------------------------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print("  SHAP CONSISTENCY ANALYSIS — INTRA-CLASS FEATURE RANK STABILITY")
    print(sep)
    print(
        f"  {'Attack Class':<22} {'n':>5} {'Mean rho':>9} {'Std rho':>8} "
        f"{'Top-3 Agree':>12}  Consensus Top-5 Features"
    )
    print("  " + "-" * 110)

    rng = np.random.default_rng(RANDOM_STATE)
    all_class_results: dict[str, dict] = {}

    for attack_class in TARGET_CLASSES:
        # Find test indices for this class
        mask = labels_test == attack_class
        class_indices = np.where(mask.values)[0]

        if len(class_indices) == 0:
            logger.warning("No test samples found for class '%s' — skipping.", attack_class)
            continue

        n_avail = len(class_indices)
        n_sample = min(args.samples, n_avail)
        sampled_idx = rng.choice(class_indices, size=n_sample, replace=False)

        X_sample = X_test.iloc[sampled_idx]
        logger.info("%-22s %d / %d samples → running SHAP...",
                    attack_class, n_sample, n_avail)

        # Run SHAP via XAIExplainer (handles all format variations)
        results = explainer.explain(X_sample)

        # Build |SHAP| matrix: (n_sample, 60)
        abs_shap = np.vstack([
            np.abs(r.raw_shap_values) if r.raw_shap_values is not None
            else np.zeros(len(feature_names))
            for r in results
        ])

        mean_rho, std_rho, rho_vals = _pairwise_spearman(abs_shap)
        top3_agree = _top3_agreement(abs_shap, feature_names)
        consensus_feats = _consensus_top_features(abs_shap, feature_names, top_n=5)

        feats_str = ", ".join(consensus_feats)
        top3_pct = f"{top3_agree * 100:.1f}%" if not np.isnan(top3_agree) else "N/A"
        mean_rho_str = f"{mean_rho:.4f}" if not np.isnan(mean_rho) else "N/A"
        std_rho_str = f"{std_rho:.4f}" if not np.isnan(std_rho) else "N/A"

        print(
            f"  {attack_class:<22} {n_sample:>5} {mean_rho_str:>9} {std_rho_str:>8} "
            f"{top3_pct:>12}  {feats_str}"
        )

        # Distribution buckets for JSON
        rho_distribution = {}
        if len(rho_vals) > 0:
            rho_distribution = {
                "≥0.9": int((rho_vals >= 0.9).sum()),
                "0.7–0.9": int(((rho_vals >= 0.7) & (rho_vals < 0.9)).sum()),
                "0.5–0.7": int(((rho_vals >= 0.5) & (rho_vals < 0.7)).sum()),
                "<0.5": int((rho_vals < 0.5).sum()),
            }

        all_class_results[attack_class] = {
            "n_available_in_test": n_avail,
            "n_sampled": n_sample,
            "mean_spearman_rho": round(mean_rho, 6) if not np.isnan(mean_rho) else None,
            "std_spearman_rho": round(std_rho, 6) if not np.isnan(std_rho) else None,
            "n_pairs": len(rho_vals),
            "top3_agreement_rate": round(top3_agree, 6) if not np.isnan(top3_agree) else None,
            "consensus_top5_features": consensus_feats,
            "rho_distribution": rho_distribution,
        }

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    valid_rhos = [
        v["mean_spearman_rho"]
        for v in all_class_results.values()
        if v["mean_spearman_rho"] is not None
    ]
    overall_mean_rho = float(np.mean(valid_rhos)) if valid_rhos else float("nan")
    valid_agrees = [
        v["top3_agreement_rate"]
        for v in all_class_results.values()
        if v["top3_agreement_rate"] is not None
    ]
    overall_top3 = float(np.mean(valid_agrees)) if valid_agrees else float("nan")

    print(f"\n{sep}")
    print(f"  Overall mean Spearman rho across all classes : {overall_mean_rho:.4f}")
    print(f"  Overall top-3 feature agreement rate        : {overall_top3 * 100:.1f}%")
    print(f"\n  Interpretation:")
    print(f"  rho close to 1.0 => model consistently uses same features for this attack type")
    print(f"  rho close to 0.0 => feature importance varies widely between instances")
    print(f"  Top-3 agree % => fraction of sample pairs sharing >=2 of their top-3 features")
    print(sep)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out = {
        "description": (
            "Pairwise Spearman rank correlation of absolute SHAP value vectors "
            "within each attack class. High rho = consistent feature attribution."
        ),
        "methodology": {
            "n_samples_per_class": args.samples,
            "test_size_fraction": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "data_source": source,
            "metric": "Spearman rho of |SHAP| vectors (60-feature, attack-class SHAP)",
            "secondary_metric": "Top-3 feature agreement: fraction of pairs sharing ≥2 top-3 features",
        },
        "overall": {
            "mean_spearman_rho": round(overall_mean_rho, 6),
            "mean_top3_agreement": round(overall_top3, 6),
        },
        "per_class": all_class_results,
    }

    out_path = args.model_dir / "shap_consistency_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
