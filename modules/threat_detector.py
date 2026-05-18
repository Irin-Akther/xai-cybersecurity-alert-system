"""
Threat Detector — Random Forest classifier trained on CICIDS2017 network flow features.

Loads a real CICIDS2017 CSV from data/ if present; otherwise generates a synthetic
dataset with identical feature schema for development and demonstration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)

# Canonical CICIDS2017 feature columns used for training.
# Strips leading/trailing whitespace from real CSV headers.
FEATURE_COLS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

LABEL_COL = "Label"

ATTACK_LABELS = [
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack – Brute Force",
    "Web Attack – XSS",
    "Web Attack – Sql Injection",
    "Infiltration",
    "Heartbleed",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"


def _load_csv(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load the first CSV found in data_dir that contains CICIDS2017 columns."""
    for csv_file in sorted(data_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            df.columns = df.columns.str.strip()
            if LABEL_COL in df.columns:
                logger.info("Loaded dataset: %s (%d rows)", csv_file.name, len(df))
                return df
        except Exception as exc:
            logger.warning("Could not read %s: %s", csv_file, exc)
    return None


def _generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic network flow data matching CICIDS2017 schema."""
    rng = np.random.default_rng(random_state)
    n_benign = int(n_samples * 0.6)
    n_attack = n_samples - n_benign

    def benign_flows(n: int) -> dict:
        return {
            "Flow Duration": rng.integers(1000, 5_000_000, n),
            "Total Fwd Packets": rng.integers(1, 50, n),
            "Total Backward Packets": rng.integers(1, 50, n),
            "Total Length of Fwd Packets": rng.integers(20, 5000, n),
            "Total Length of Bwd Packets": rng.integers(20, 5000, n),
            "Fwd Packet Length Max": rng.integers(20, 1500, n),
            "Fwd Packet Length Min": rng.integers(0, 100, n),
            "Fwd Packet Length Mean": rng.uniform(20, 800, n),
            "Fwd Packet Length Std": rng.uniform(0, 300, n),
            "Bwd Packet Length Max": rng.integers(20, 1500, n),
            "Bwd Packet Length Min": rng.integers(0, 100, n),
            "Bwd Packet Length Mean": rng.uniform(20, 800, n),
            "Bwd Packet Length Std": rng.uniform(0, 300, n),
            "Flow Bytes/s": rng.uniform(100, 1_000_000, n),
            "Flow Packets/s": rng.uniform(1, 10_000, n),
            "Flow IAT Mean": rng.uniform(100, 100_000, n),
            "Flow IAT Std": rng.uniform(10, 50_000, n),
            "Flow IAT Max": rng.integers(1000, 500_000, n),
            "Flow IAT Min": rng.integers(0, 1000, n),
            "Fwd IAT Total": rng.integers(0, 1_000_000, n),
            "Fwd IAT Mean": rng.uniform(0, 200_000, n),
            "Bwd IAT Total": rng.integers(0, 1_000_000, n),
            "Bwd IAT Mean": rng.uniform(0, 200_000, n),
            "Fwd PSH Flags": rng.integers(0, 2, n),
            "Bwd PSH Flags": rng.integers(0, 2, n),
            "Fwd Header Length": rng.integers(20, 60, n) * rng.integers(1, 20, n),
            "Bwd Header Length": rng.integers(20, 60, n) * rng.integers(1, 20, n),
            "Fwd Packets/s": rng.uniform(0.5, 5000, n),
            "Bwd Packets/s": rng.uniform(0.5, 5000, n),
            "Min Packet Length": rng.integers(0, 100, n),
            "Max Packet Length": rng.integers(100, 1500, n),
            "Packet Length Mean": rng.uniform(50, 800, n),
            "Packet Length Std": rng.uniform(10, 400, n),
            "Packet Length Variance": rng.uniform(100, 160_000, n),
            "FIN Flag Count": rng.integers(0, 2, n),
            "SYN Flag Count": rng.integers(0, 2, n),
            "RST Flag Count": rng.integers(0, 1, n),
            "PSH Flag Count": rng.integers(0, 3, n),
            "ACK Flag Count": rng.integers(0, 10, n),
            "URG Flag Count": rng.integers(0, 1, n),
            "Down/Up Ratio": rng.uniform(0.1, 5, n),
            "Average Packet Size": rng.uniform(50, 800, n),
            "Avg Fwd Segment Size": rng.uniform(20, 800, n),
            "Avg Bwd Segment Size": rng.uniform(20, 800, n),
            "Subflow Fwd Packets": rng.integers(1, 50, n),
            "Subflow Fwd Bytes": rng.integers(20, 5000, n),
            "Subflow Bwd Packets": rng.integers(1, 50, n),
            "Subflow Bwd Bytes": rng.integers(20, 5000, n),
            "Init_Win_bytes_forward": rng.integers(0, 65535, n),
            "Init_Win_bytes_backward": rng.integers(0, 65535, n),
            "act_data_pkt_fwd": rng.integers(1, 30, n),
            "min_seg_size_forward": rng.integers(20, 60, n),
            "Active Mean": rng.uniform(0, 100_000, n),
            "Active Std": rng.uniform(0, 50_000, n),
            "Active Max": rng.integers(0, 200_000, n),
            "Active Min": rng.integers(0, 50_000, n),
            "Idle Mean": rng.uniform(0, 1_000_000, n),
            "Idle Std": rng.uniform(0, 500_000, n),
            "Idle Max": rng.integers(0, 2_000_000, n),
            "Idle Min": rng.integers(0, 500_000, n),
        }

    def attack_flows(n: int) -> dict:
        d = benign_flows(n)
        # Attacks exhibit high packet rates, large volumes, unusual flag counts
        d["Flow Packets/s"] = rng.uniform(10_000, 1_000_000, n)
        d["Flow Bytes/s"] = rng.uniform(1_000_000, 50_000_000, n)
        d["Total Fwd Packets"] = rng.integers(100, 10_000, n)
        d["SYN Flag Count"] = rng.integers(1, 10, n)
        d["RST Flag Count"] = rng.integers(1, 5, n)
        d["Flow IAT Mean"] = rng.uniform(0, 500, n)
        d["Flow Duration"] = rng.integers(1, 50_000, n)
        return d

    benign_df = pd.DataFrame(benign_flows(n_benign))
    benign_df[LABEL_COL] = "BENIGN"

    attack_labels = rng.choice(ATTACK_LABELS, size=n_attack)
    attack_df = pd.DataFrame(attack_flows(n_attack))
    attack_df[LABEL_COL] = attack_labels

    df = pd.concat([benign_df, attack_df], ignore_index=True).sample(
        frac=1, random_state=random_state
    )
    logger.info("Generated synthetic dataset: %d rows", len(df))
    return df


def _preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean data, drop infinite/NaN values, return (X, y_binary)."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        logger.warning("Missing %d feature columns; filling with 0: %s", len(missing), missing)
        for col in missing:
            df[col] = 0

    X = df[FEATURE_COLS].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    X = X.clip(lower=-1e15, upper=1e15)

    y_raw = df[LABEL_COL].str.strip()
    y_binary = (y_raw != "BENIGN").astype(int)
    return X, y_binary


class ThreatDetector:
    """Random Forest-based network intrusion detector with CICIDS2017 feature schema."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: list[str] = FEATURE_COLS
        self._is_trained = False

    def train(self, data_dir: Optional[Path] = None) -> dict:
        """Train on real CICIDS2017 CSV or synthetic fallback. Returns metrics dict."""
        data_dir = data_dir or DATA_DIR
        df = _load_csv(data_dir)
        if df is None:
            logger.warning("No CICIDS2017 CSV found in %s — using synthetic data.", data_dir)
            df = _generate_synthetic_data()

        X, y = _preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)
        self._is_trained = True

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"], output_dict=True)
        logger.info("Training complete — accuracy: %.4f", accuracy)
        return {"accuracy": accuracy, "report": report}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0=benign, 1=attack) for a feature DataFrame."""
        self._require_trained()
        X_clean = X[self.feature_names].copy()
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_clean.fillna(0, inplace=True)
        return self.model.predict(X_clean)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability scores for [benign, attack] classes."""
        self._require_trained()
        X_clean = X[self.feature_names].copy()
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_clean.fillna(0, inplace=True)
        return self.model.predict_proba(X_clean)

    def predict_single(self, features: dict) -> tuple[int, float]:
        """Predict a single flow given a feature dict. Returns (label, confidence)."""
        row = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in row.columns:
                row[col] = 0
        pred = self.predict(row)[0]
        proba = self.predict_proba(row)[0]
        confidence = float(proba[pred])
        return int(pred), confidence

    def save(self, model_dir: Optional[Path] = None):
        """Persist the trained model to disk."""
        self._require_trained()
        model_dir = model_dir or MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_dir / "random_forest.joblib")
        logger.info("Model saved to %s", model_dir)

    @classmethod
    def load(cls, model_dir: Optional[Path] = None) -> "ThreatDetector":
        """Load a previously trained model from disk."""
        model_dir = model_dir or MODEL_DIR
        path = model_dir / "random_forest.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Run train.py first.")
        instance = cls()
        instance.model = joblib.load(path)
        instance._is_trained = True
        logger.info("Model loaded from %s", path)
        return instance

    def _require_trained(self):
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call .train() or .load() first.")
