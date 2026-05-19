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
    value: float          # actual feature value from the flow
    shap_value: float     # SHAP contribution toward attack class
    direction: str        # "increases_risk" | "decreases_risk"

    def as_dict(self) -> dict:
        return {
            "feature": self.name,
            "value": round(self.value, 4),
            "shap_value": round(self.shap_value, 4),
            "direction": self.direction,
        }


@dataclass
class ExplanationResult:
    predicted_label: int          # 0 = BENIGN, 1 = ATTACK
    confidence: float             # probability of the predicted class
    base_value: float             # SHAP expected value
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
    """SHAP-based explainer for a trained RandomForestClassifier.

    Wraps shap.TreeExplainer for fast, exact Shapley values on tree models.
    The explainer is initialised once and reused across predictions.
    """

    def __init__(self, detector, top_n: int = 10):
        """
        Args:
            detector: A trained ThreatDetector instance.
            top_n: Number of top contributing features to return per explanation.
        """
        self._detector = detector
        self.top_n = top_n
        self._explainer: Optional[shap.TreeExplainer] = None

    def _get_explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            logger.info("Initialising SHAP TreeExplainer (one-time setup)…")
            self._explainer = shap.TreeExplainer(
                self._detector.model,
                feature_perturbation="tree_path_dependent",
            )
        return self._explainer

    def explain(self, X: pd.DataFrame) -> list[ExplanationResult]:
        """Return SHAP explanations for every row in X."""
        X_clean = self._prepare(X)
        probas = self._detector.predict_proba(X)
        preds = self._detector.predict(X)
        feature_names = self._detector.feature_names
        n_features = len(feature_names)

        # --- SHAP computation with full fallback ---
        attack_shaps = None
        base_value = 0.5
        try:
            explainer = self._get_explainer()
            try:
                shap_values = explainer.shap_values(X_clean, check_additivity=False)
            except TypeError:
                shap_values = explainer.shap_values(X_clean)
            attack_shaps = self._extract_attack_shaps(shap_values, X_clean)
            ev = explainer.expected_value
            if hasattr(ev, "__len__"):
                base_value = float(np.ravel(np.asarray(ev, dtype=float))[1])
            else:
                base_value = float(ev)
        except Exception as exc:
            logger.warning("SHAP failed (%s); using feature importances as fallback.", exc)

        if attack_shaps is None:
            attack_shaps = self._fallback_importances(X_clean)

        results = []
        for i in range(len(X_clean)):
            try:
                row_shap = np.ravel(np.asarray(attack_shaps[i], dtype=float))
            except Exception:
                row_shap = np.zeros(n_features)
            if len(row_shap) != n_features:
                row_shap = np.zeros(n_features)

            row_vals = X_clean.iloc[i].values

            contributions = []
            for fname, fval, sval in zip(feature_names, row_vals, row_shap):
                try:
                    sv = float(sval)
                except (TypeError, ValueError):
                    sv = 0.0
                try:
                    fv = float(fval)
                except (TypeError, ValueError):
                    fv = 0.0
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
            confidence = float(probas[i][pred])
            results.append(
                ExplanationResult(
                    predicted_label=pred,
                    confidence=confidence,
                    base_value=base_value,
                    top_features=contributions[: self.top_n],
                    raw_shap_values=row_shap,
                )
            )

        return results

    def explain_single(self, features: dict) -> ExplanationResult:
        """Explain a single flow given a feature dictionary."""
        row = pd.DataFrame([features])
        for col in self._detector.feature_names:
            if col not in row.columns:
                row[col] = 0.0
        return self.explain(row)[0]

    def _fallback_importances(self, X_clean: pd.DataFrame) -> np.ndarray:
        """Return global feature importances shaped as (n_samples, n_features) fallback."""
        importances = np.array(self._detector.model.feature_importances_, dtype=float)
        return np.tile(importances, (len(X_clean), 1))

    def _extract_attack_shaps(self, shap_values, X_clean: pd.DataFrame) -> np.ndarray:
        """Convert any SHAP output format to a plain (n_samples, n_features) float array."""
        n_samples = len(X_clean)
        n_features = len(self._detector.feature_names)
        try:
            # Unwrap list [benign, attack]
            if isinstance(shap_values, list):
                raw = shap_values[1]
            else:
                raw = shap_values

            # Unwrap SHAP Explanation objects (potentially nested)
            for _ in range(3):
                if hasattr(raw, "values"):
                    raw = raw.values
                else:
                    break

            arr = np.asarray(raw, dtype=float)

            if arr.ndim == 3:        # (n_samples, n_features, n_classes)
                arr = arr[:, :, 1]
            elif arr.ndim == 1:      # (n_features,) — single sample squeezed
                arr = arr.reshape(1, n_features)
            # ndim == 2 is already (n_samples, n_features)

            if arr.shape[0] != n_samples:
                arr = arr.reshape(n_samples, n_features)

            return arr
        except Exception as exc:
            logger.warning("SHAP extraction failed (%s); using zeros.", exc)
            return np.zeros((n_samples, n_features))

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()
        for col in self._detector.feature_names:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
        X_clean = X_clean[self._detector.feature_names]
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_clean.fillna(0, inplace=True)
        return X_clean

    def shap_summary_data(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Return (shap_values_matrix, feature_names) suitable for shap.summary_plot."""
        explainer = self._get_explainer()
        X_clean = self._prepare(X)
        shap_values = explainer.shap_values(X_clean)
        attack_shaps = self._extract_attack_shaps(shap_values, X_clean)
        return attack_shaps, self._detector.feature_names
