"""
NLG Module — Reading-level-adaptive natural language explanations via local Ollama LLM.

Privacy-preserving design: all inference runs on-device through Ollama.
No network traffic data is sent to external services.

Falls back to a template-based explanation when Ollama is unavailable,
ensuring the system degrades gracefully.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Optional

import requests

from modules.user_profiler import LiteracyLevel, UserProfile
from modules.xai_explainer import ExplanationResult

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"        # mistral or llama3 — change to match installed model
REQUEST_TIMEOUT = 60             # seconds


def _build_prompt(explanation: ExplanationResult, profile: UserProfile) -> str:
    """Construct a reading-level-targeted prompt for the LLM."""
    label = "a network attack" if explanation.predicted_label == 1 else "normal network traffic"
    confidence_pct = f"{explanation.confidence * 100:.1f}%"

    top_feature_lines = "\n".join(
        f"  - {fc.name}: {fc.value:.2f} (SHAP={fc.shap_value:+.4f}, {fc.direction.replace('_', ' ')})"
        for fc in explanation.top_features[:5]
    )

    level_instructions = {
        LiteracyLevel.HOME: (
            "Explain this to a non-technical home user. Use simple, everyday language. "
            "Avoid technical jargon. Focus on what the threat means for the user personally "
            "and what simple action they should take. Keep it under 80 words."
        ),
        LiteracyLevel.SMB: (
            "Explain this to an IT support professional. Include which network behaviour "
            "triggered the alert and suggest a practical remediation step. "
            "Use clear technical language without deep security jargon. Under 120 words."
        ),
        LiteracyLevel.ADMIN: (
            "Provide a full technical analysis for a SOC analyst. Include the top SHAP "
            "features driving the classification, the likely attack vector, and recommended "
            "immediate containment steps. Reference MITRE ATT&CK tactics if applicable. "
            "Up to 200 words."
        ),
    }

    instruction = level_instructions[profile.level]

    return textwrap.dedent(f"""
        You are an expert cybersecurity AI assistant embedded in an Explainable AI (XAI) threat detection system.

        Detection result:
        - Classification: {label.upper()}
        - Model confidence: {confidence_pct}
        - Top contributing network flow features:
        {top_feature_lines}

        Task: {instruction}

        Write only the explanation — no headers, no markdown, no preamble.
    """).strip()


def _template_fallback(explanation: ExplanationResult, profile: UserProfile) -> str:
    """Rule-based template explanation used when Ollama is unavailable."""
    label_text = "ATTACK" if explanation.predicted_label == 1 else "BENIGN"
    confidence_pct = f"{explanation.confidence * 100:.1f}%"
    top = explanation.top_features[:3]
    top_str = ", ".join(f"{f.name} ({f.direction.replace('_', ' ')})" for f in top)

    if profile.level == LiteracyLevel.HOME:
        if explanation.predicted_label == 1:
            return (
                f"Our system detected unusual activity on your network with {confidence_pct} confidence. "
                "This might be someone trying to access your device without permission. "
                "We recommend restarting your router and checking that your security software is up to date."
            )
        else:
            return (
                f"Your network activity looks normal ({confidence_pct} confidence). "
                "No immediate action is needed."
            )

    if profile.level == LiteracyLevel.SMB:
        if explanation.predicted_label == 1:
            return (
                f"Threat detected ({confidence_pct} confidence). "
                f"Key indicators: {top_str}. "
                "Recommended action: isolate the affected host, review firewall rules, "
                "and check for unauthorised services or open ports."
            )
        else:
            return (
                f"Traffic classified as benign ({confidence_pct} confidence). "
                "No immediate action required. Continue standard monitoring."
            )

    # ADMIN
    if explanation.predicted_label == 1:
        shap_details = "; ".join(
            f"{f.name}={f.value:.2f} (SHAP {f.shap_value:+.4f})"
            for f in top
        )
        return (
            f"Classification: {label_text} | Confidence: {confidence_pct}\n"
            f"Top SHAP drivers: {shap_details}\n"
            "Recommended: correlate with SIEM events, check for lateral movement indicators, "
            "apply network segmentation, and initiate IR playbook if persistence is suspected."
        )
    else:
        return (
            f"Classification: {label_text} | Confidence: {confidence_pct}. "
            "No anomalous indicators in top SHAP features. Continue baseline monitoring."
        )


class NLGModule:
    """Generates reading-level-adaptive threat explanations using a local Ollama LLM.

    All inference is performed on-device; no data leaves the local machine.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._ollama_available: Optional[bool] = None

    def _check_ollama(self) -> bool:
        """Check whether Ollama is running and the model is available."""
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                available_models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
                if self.model in available_models:
                    logger.info("Ollama is running with model '%s'.", self.model)
                    self._ollama_available = True
                    return True
                else:
                    logger.warning(
                        "Ollama is running but model '%s' is not installed. "
                        "Run: ollama pull %s",
                        self.model,
                        self.model,
                    )
            self._ollama_available = False
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable at %s. Using template fallback.", self.base_url)
            self._ollama_available = False
        return self._ollama_available

    def generate(self, explanation: ExplanationResult, profile: UserProfile) -> str:
        """Generate a natural language explanation for the given alert and user profile.

        Uses Ollama if available; otherwise returns a template-based explanation.

        Args:
            explanation: SHAP ExplanationResult from XAIExplainer.
            profile: UserProfile determining reading level.

        Returns:
            Human-readable explanation string.
        """
        if self._check_ollama():
            try:
                return self._call_ollama(explanation, profile)
            except Exception as exc:
                logger.error("Ollama inference failed: %s — falling back to template.", exc)

        return _template_fallback(explanation, profile)

    def _call_ollama(self, explanation: ExplanationResult, profile: UserProfile) -> str:
        prompt = _build_prompt(explanation, profile)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 300},
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def list_available_models(self) -> list[str]:
        """Return model names available in the local Ollama installation."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    def set_model(self, model_name: str):
        """Switch to a different Ollama model."""
        self.model = model_name
        self._ollama_available = None   # re-check on next generate call
