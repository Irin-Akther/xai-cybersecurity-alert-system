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
        """Return SHAP explanations for every row in X.

        Args:
            X: DataFrame with CICIDS2017 feature columns.

        Returns:
            List of ExplanationResult, one per row.
        """
        explainer = self._get_explainer()
        X_clean = self._prepare(X)

        shap_values = explainer.shap_values(X_clean)
        # Robust extraction: handles list, 3-D array (n_samples, n_features, n_classes),
        # or a SHAP Explanation object returned by newer shap versions.
        if isinstance(shap_values, list):
            attack_shaps = np.array(shap_values[1])
        elif hasattr(shap_values, "values"):
            v = np.array(shap_values.values)
            attack_shaps = v[:, :, 1] if v.ndim == 3 else v
        else:
            sv = np.array(shap_values)
            attack_shaps = sv[:, :, 1] if sv.ndim == 3 else sv

        base_value = explainer.expected_value
        if hasattr(base_value, "__len__"):
            base_value = float(np.ravel(base_value)[1])
        else:
            base_value = float(base_value)

        probas = self._detector.predict_proba(X)
        preds = self._detector.predict(X)

        results = []
        for i in range(len(X_clean)):
            row_shap = attack_shaps[i]
            row_vals = X_clean.iloc[i].values
            feature_names = self._detector.feature_names

            contributions = []
            for fname, fval, sval in zip(feature_names, row_vals, row_shap):
                # ravel guards against SHAP returning a 1-element array instead of scalar
                shap_val = float(np.ravel(np.asarray(sval))[0])
                contributions.append(
                    FeatureContribution(
                        name=fname,
                        value=float(np.ravel(np.asarray(fval))[0]),
                        shap_value=shap_val,
                        direction="increases_risk" if shap_val > 0 else "decreases_risk",
                    )
                )

            # Sort by absolute SHAP magnitude descending
            contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
            top = contributions[: self.top_n]

            pred = int(preds[i])
            confidence = float(probas[i][pred])

            results.append(
                ExplanationResult(
                    predicted_label=pred,
                    confidence=confidence,
                    base_value=base_value,
                    top_features=top,
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
        if isinstance(shap_values, list):
            attack_shaps = np.array(shap_values[1])
        elif hasattr(shap_values, "values"):
            v = np.array(shap_values.values)
            attack_shaps = v[:, :, 1] if v.ndim == 3 else v
        else:
            sv = np.array(shap_values)
            attack_shaps = sv[:, :, 1] if sv.ndim == 3 else sv
        return attack_shaps, self._detector.feature_names
