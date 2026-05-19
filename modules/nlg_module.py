"""
NLG Module — Persona-aware, reading-level-adaptive natural language explanations via local Ollama LLM.

Privacy-preserving design: all inference runs on-device through Ollama.
No network traffic data is sent to external services.

Falls back to a template-based explanation when Ollama is unavailable.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Optional

import requests

from modules.user_profiler import LiteracyLevel, Persona, UserProfile
from modules.xai_explainer import ExplanationResult

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"
REQUEST_TIMEOUT = 60


def _build_prompt(explanation: ExplanationResult, profile: UserProfile) -> str:
    label = "a network attack" if explanation.predicted_label == 1 else "normal network traffic"
    confidence_pct = f"{explanation.confidence * 100:.1f}%"

    top_feature_lines = "\n".join(
        f"  - {fc.name}: {fc.value:.2f} (SHAP={fc.shap_value:+.4f}, {fc.direction.replace('_', ' ')})"
        for fc in explanation.top_features[:5]
    )

    return textwrap.dedent(f"""
        You are an expert cybersecurity AI assistant inside an Explainable AI (XAI) threat detection system.

        Detection result:
        - Classification: {label.upper()}
        - Model confidence: {confidence_pct}
        - Top contributing network flow features:
        {top_feature_lines}

        The person receiving this alert is: {profile.persona.value}

        Task: {profile.tone_instruction}

        Write only the explanation — no headers, no markdown, no preamble.
    """).strip()


def _template_fallback(explanation: ExplanationResult, profile: UserProfile) -> str:
    """Rule-based template explanation used when Ollama is unavailable."""
    confidence_pct = f"{explanation.confidence * 100:.1f}%"
    is_attack = explanation.predicted_label == 1
    top = explanation.top_features[:3]
    top_str = ", ".join(f"{f.name} ({f.direction.replace('_', ' ')})" for f in top)

    persona = profile.persona

    # --- HOME-level personas ---
    if persona == Persona.KID:
        if is_attack:
            return (
                f"Uh oh! Something sneaky is happening on your internet ({confidence_pct} sure). "
                "It's like a stranger trying to open your front door. "
                "Tell a grown-up right away and don't click anything unusual!"
            )
        return "Everything looks safe! Your internet is working normally. Keep it up! 🎉"

    if persona == Persona.TEENAGER:
        if is_attack:
            return (
                f"Heads up — looks like something sketchy is going on with your network "
                f"({confidence_pct} confidence). Someone might be trying to hack in. "
                "Disconnect from Wi-Fi and let someone know."
            )
        return f"All clear — your network traffic looks normal ({confidence_pct} confidence). No action needed."

    if persona == Persona.HOUSEWIFE:
        if is_attack:
            return (
                f"Our system detected something unusual on your home internet ({confidence_pct} sure). "
                "Think of it like an alarm going off at your front door. "
                "Please restart your router and let your family know to avoid clicking strange links."
            )
        return "Your home internet looks safe right now. No action needed."

    if persona == Persona.CASHIER:
        if is_attack:
            return (
                f"Security alert on your work system ({confidence_pct} confident). "
                "Stop what you're doing and call your manager or IT support immediately. "
                "Don't enter any passwords until they say it's safe."
            )
        return "System looks normal. No action needed — continue with your work."

    if persona == Persona.GENERAL_EMPLOYEE:
        if is_attack:
            return (
                f"A potential security threat was detected on your network ({confidence_pct} confidence). "
                "Please stop using sensitive systems, disconnect from the network if possible, "
                "and contact your IT helpdesk immediately."
            )
        return f"Your network activity appears normal ({confidence_pct} confidence). No action required."

    # --- SMB-level personas ---
    if persona == Persona.BUSINESS_OWNER:
        if is_attack:
            return (
                f"Security threat detected ({confidence_pct} confidence). "
                "Your business data or operations may be at risk. "
                f"Key signals: {top_str}. "
                "Immediate action: isolate the affected device, contact your IT provider, "
                "and consider notifying customers if data may be involved."
            )
        return f"Network traffic looks normal ({confidence_pct} confidence). No business impact detected."

    if persona == Persona.STUDENT:
        if is_attack:
            top_learning = explanation.top_features[0] if explanation.top_features else None
            learning_note = (
                f" The strongest indicator was **{top_learning.name}** "
                f"(SHAP={top_learning.shap_value:+.4f}), which suggests abnormal flow behaviour."
                if top_learning else ""
            )
            return (
                f"Attack detected ({confidence_pct} confidence).{learning_note} "
                f"Key features: {top_str}. "
                "Learning tip: this pattern is consistent with volumetric or scanning attacks "
                "where flow statistics deviate significantly from baseline."
            )
        return (
            f"Traffic classified as benign ({confidence_pct} confidence). "
            "The model found no significant anomalies in the flow features."
        )

    if persona == Persona.EXECUTIVE:
        if is_attack:
            return (
                f"RISK ALERT — {confidence_pct} confidence. "
                "A network intrusion attempt has been detected that may impact operations or data integrity. "
                "Recommended action: authorise your IT team to isolate the affected system immediately."
            )
        return f"No threat detected ({confidence_pct} confidence). Operations are unaffected."

    # --- ADMIN-level personas ---
    if persona == Persona.COMPLIANCE:
        label_text = "ATTACK" if is_attack else "BENIGN"
        if is_attack:
            shap_details = "; ".join(
                f"{f.name}={f.value:.2f} (SHAP {f.shap_value:+.4f})"
                for f in top
            )
            return (
                f"Classification: {label_text} | Confidence: {confidence_pct}\n"
                f"Top indicators: {shap_details}\n"
                "Compliance note: this event may trigger notification obligations under "
                "NIST IR 800-61, ISO 27001 A.16, or GDPR Article 33 depending on data scope. "
                "Ensure incident is logged in your audit trail with timestamp and evidence preserved."
            )
        return (
            f"Classification: {label_text} | Confidence: {confidence_pct}. "
            "No compliance-relevant indicators detected. Log for audit completeness."
        )

    # Security Analyst (default ADMIN fallback)
    label_text = "ATTACK" if is_attack else "BENIGN"
    if is_attack:
        shap_details = "; ".join(
            f"{f.name}={f.value:.2f} (SHAP {f.shap_value:+.4f})"
            for f in top
        )
        return (
            f"Classification: {label_text} | Confidence: {confidence_pct}\n"
            f"Top SHAP drivers: {shap_details}\n"
            "Recommended: initiate IR playbook, correlate with SIEM/EDR, "
            "check for lateral movement, apply network segmentation."
        )
    return (
        f"Classification: {label_text} | Confidence: {confidence_pct}. "
        "No anomalous indicators. Continue baseline monitoring."
    )


class NLGModule:
    """Generates persona-adaptive threat explanations using a local Ollama LLM."""

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
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                available_models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
                if self.model in available_models:
                    self._ollama_available = True
                    return True
                logger.warning("Ollama running but model '%s' not installed. Run: ollama pull %s", self.model, self.model)
            self._ollama_available = False
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable at %s. Using template fallback.", self.base_url)
            self._ollama_available = False
        return self._ollama_available

    def generate(self, explanation: ExplanationResult, profile: UserProfile) -> str:
        """Generate a natural language explanation for the given alert and user profile."""
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
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    def set_model(self, model_name: str):
        self.model = model_name
        self._ollama_available = None
