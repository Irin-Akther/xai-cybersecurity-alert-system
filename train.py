"""
Training script for the XAI Cybersecurity Alert System.

Usage:
    python train.py                         # train on synthetic CICIDS2017-schema data
    python train.py --data data/your.csv    # train on a real CICIDS2017 CSV
    python train.py --estimators 200        # custom Random Forest parameters
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

from modules.threat_detector import ThreatDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XAI Threat Detection model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to a CICIDS2017 CSV file (optional; synthetic data used if omitted).",
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

    data_dir = args.data.parent if args.data else PROJECT_ROOT / "data"
    if args.data and not args.data.exists():
        logger.error("Data file not found: %s", args.data)
        sys.exit(1)

    logger.info("Initialising ThreatDetector (n_estimators=%d, max_depth=%d)",
                args.estimators, args.max_depth)

    detector = ThreatDetector(
        n_estimators=args.estimators,
        max_depth=args.max_depth,
    )

    logger.info("Starting training…")
    metrics = detector.train(data_dir=data_dir)

    logger.info("Training complete.")
    logger.info("  Accuracy : %.4f", metrics["accuracy"])

    report = metrics["report"]
    for label in ["BENIGN", "ATTACK"]:
        if label in report:
            r = report[label]
            logger.info(
                "  %-8s  precision=%.3f  recall=%.3f  f1=%.3f  support=%d",
                label,
                r["precision"],
                r["recall"],
                r["f1-score"],
                int(r["support"]),
            )

    logger.info("Saving model to %s", args.output)
    detector.save(args.output)

    metrics_path = args.output / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "accuracy": metrics["accuracy"],
        "report": {
            k: v for k, v in metrics["report"].items()
            if isinstance(v, dict)
        },
    }
    metrics_path.write_text(json.dumps(serialisable, indent=2))
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
