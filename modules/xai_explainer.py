"""
XAI Explainer — SHAP TreeExplainer for Random Forest feature attribution.

Generates per-prediction SHAP values and surfaces the top-N features that
most influenced the threat classification decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    name: str
    value: float
    shap_value: float
    direction: str

    def as_dict(self) -> dict:
        return {
            "feature": self.name,
            "value": round(self.value, 4),
            "shap_value": round(self.shap_value, 4),
            "direction": self.direction,
        }


@dataclass
class ExplanationResult:
    predicted_label: int
    confidence: float
    base_value: float
    top_features: list[FeatureContribution] = field(default_factory=list)
    raw_shap_values: Optional[np.ndarray] = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "predicted_label": self.predicted_label,
            "label_text": "ATTACK" if self.predicted_label == 1 else "BENIGN",
            "confidence": round(self.confidence, 4),
            "base_value": round(self.base_value, 4),
            "top_features": [f.as_dict() for f in self.top_features],
        }


class XAIExplainer:
    """SHAP-based explainer for a trained RandomForestClassifier."""

    def __init__(self, detector, top_n: int = 10):
        self._detector = detector
        self.top_n = top_n
        self._explainer: Optional[shap.TreeExplainer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, X: pd.DataFrame) -> list[ExplanationResult]:
        """Return SHAP explanations for every row in X."""
        X_clean = self._prepare(X)
        feature_names = list(self._detector.feature_names)
        n = len(X_clean)
        k = len(feature_names)

        # Predictions — these always work
        preds = np.asarray(self._detector.predict(X), dtype=int).ravel()
        probas = np.asarray(self._detector.predict_proba(X), dtype=float)

        # SHAP matrix: guaranteed (n_samples, n_features) float64
        shap_matrix = self._safe_shap(X_clean, n, k)
        base_val = self._safe_base_value()

        # Use plain numpy arrays; avoids any pandas indexing quirks
        X_arr = np.asarray(X_clean.values, dtype=float)

        results = []
        for i in range(n):
            row_vals = X_arr[i]
            row_shap = shap_matrix[i]

            contributions = []
            for j, fname in enumerate(feature_names):
                sv = float(row_shap[j])
                fv = float(row_vals[j])
                contributions.append(
                    FeatureContribution(
                        name=fname,
                        value=fv,
                        shap_value=sv,
                        direction="increases_risk" if sv > 0 else "decreases_risk",
                    )
                )
            contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

            pred = int(preds[i])
            # Guard against a 1-class probas array
            col = pred if pred < probas.shape[1] else 0
            conf = float(probas[i, col])

            results.append(
                ExplanationResult(
                    predicted_label=pred,
                    confidence=conf,
                    base_value=base_val,
                    top_features=contributions[: self.top_n],
                    raw_shap_values=row_shap,
                )
            )
        return results

    def explain_single(self, features: dict) -> ExplanationResult:
        """Explain a single flow — guaranteed never to raise."""
        try:
            feature_names = list(self._detector.feature_names)
            row = pd.DataFrame([{f: float(features.get(f, 0.0)) for f in feature_names}])
            return self.explain(row)[0]
        except Exception as exc:
            logger.error("explain_single failed: %s", exc, exc_info=True)
            return self._emergency_result(features)

    def shap_summary_data(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        X_clean = self._prepare(X)
        n, k = len(X_clean), len(self._detector.feature_names)
        shap_matrix = self._safe_shap(X_clean, n, k)
        return shap_matrix, list(self._detector.feature_names)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_shap(self, X_clean: pd.DataFrame, n_samples: int, n_features: int) -> np.ndarray:
        """Return (n_samples, n_features) float64 SHAP array; falls back to importances."""
        try:
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(
                    self._detector.model,
                    feature_perturbation="tree_path_dependent",
                )
            # Pass numpy array — avoids pandas/SHAP version incompatibilities
            X_np = np.asarray(X_clean.values, dtype=float)
            try:
                sv = self._explainer.shap_values(X_np, check_additivity=False)
            except Exception:
                sv = self._explainer.shap_values(X_np)
            return self._parse_shap(sv, n_samples, n_features)
        except Exception as exc:
            logger.warning("SHAP unavailable (%s); falling back to feature importances.", exc)
            return self._importance_matrix(n_samples, n_features)

    def _parse_shap(self, sv, n_samples: int, n_features: int) -> np.ndarray:
        """Convert any SHAP output format to (n_samples, n_features) float64."""
        try:
            # Unwrap list [class0_shaps, class1_shaps]
            if isinstance(sv, list):
                raw = sv[1]
            else:
                raw = sv
            # Unwrap SHAP Explanation objects
            while hasattr(raw, "values"):
                raw = raw.values
            arr = np.asarray(raw, dtype=float)
            # (n_samples, n_features, n_classes) → pick class 1
            if arr.ndim == 3:
                arr = arr[:, :, 1]
            # Ensure shape is exactly (n_samples, n_features)
            arr = arr.reshape(n_samples, n_features)
            return arr
        except Exception as exc:
            logger.warning("SHAP parse failed (%s); using zeros.", exc)
            return np.zeros((n_samples, n_features))

    def _importance_matrix(self, n_samples: int, n_features: int) -> np.ndarray:
        try:
            imp = np.asarray(self._detector.model.feature_importances_, dtype=float)
            return np.tile(imp, (n_samples, 1))
        except Exception:
            return np.zeros((n_samples, n_features))

    def _safe_base_value(self) -> float:
        try:
            ev = getattr(self._explainer, "expected_value", 0.5)
            if hasattr(ev, "__len__"):
                arr = np.asarray(ev, dtype=float).ravel()
                return float(arr[min(1, len(arr) - 1)])
            return float(ev)
        except Exception:
            return 0.5

    def _emergency_result(self, features: dict) -> ExplanationResult:
        """Minimal valid result when everything else fails."""
        try:
            feature_names = list(self._detector.feature_names)
            row = pd.DataFrame([{f: float(features.get(f, 0.0)) for f in feature_names}])
            pred = int(self._detector.predict(row)[0])
            probas = np.asarray(self._detector.predict_proba(row), dtype=float)
            col = pred if pred < probas.shape[1] else 0
            conf = float(probas[0, col])
        except Exception:
            pred, conf = 0, 0.5
        return ExplanationResult(
            predicted_label=pred,
            confidence=conf,
            base_value=0.5,
            top_features=[],
            raw_shap_values=None,
        )

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        feature_names = list(self._detector.feature_names)
        X_clean = X.copy()
        for col in feature_names:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
        X_clean = X_clean[feature_names]
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_clean.fillna(0, inplace=True)
        return X_clean
