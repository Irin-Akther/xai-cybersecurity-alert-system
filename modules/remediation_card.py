"""
Remediation Card — structured alert card formatter.

Combines threat detection, SHAP explanation, user-adaptive NLG text, and
actionable remediation guidance into a single structured AlertCard object.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

from modules.user_profiler import LiteracyLevel, UserProfile
from modules.xai_explainer import ExplanationResult

# ---------------------------------------------------------------------------
# Remediation action library keyed by literacy level
# ---------------------------------------------------------------------------

_REMEDIATION_ACTIONS: dict[LiteracyLevel, dict[str, list[str]]] = {
    LiteracyLevel.HOME: {
        "ATTACK": [
            "Restart your router by unplugging it for 30 seconds.",
            "Run a virus/malware scan on your device.",
            "Change your Wi-Fi password.",
            "Contact your Internet Service Provider if the problem continues.",
        ],
        "BENIGN": [
            "No action needed — your network looks healthy.",
        ],
    },
    LiteracyLevel.SMB: {
        "ATTACK": [
            "Isolate the affected host from the network immediately.",
            "Review firewall rules and block suspicious source IP addresses.",
            "Inspect open ports and disable any unauthorised services.",
            "Check for abnormal user accounts or privilege escalations.",
            "Apply latest OS and application security patches.",
            "Escalate to your security provider or MSSP if needed.",
        ],
        "BENIGN": [
            "Traffic is within normal parameters.",
            "Maintain standard monitoring cadence.",
        ],
    },
    LiteracyLevel.ADMIN: {
        "ATTACK": [
            "Initiate IR playbook: contain → eradicate → recover.",
            "Capture memory dump and network PCAP for forensic analysis.",
            "Correlate SHAP-flagged features against SIEM/EDR telemetry.",
            "Check for lateral movement using Windows Event ID 4624/4625.",
            "Review MITRE ATT&CK mapping for identified attack pattern.",
            "Block malicious IPs/domains at perimeter and DNS level.",
            "Preserve evidence chain-of-custody before remediation.",
            "Update detection signatures in IDS/IPS based on this incident.",
        ],
        "BENIGN": [
            "No indicators of compromise detected.",
            "Continue threat hunting on lower-confidence flows.",
            "Verify baseline model still reflects current network environment.",
        ],
    },
}

# MITRE ATT&CK tactic hints per attack type (best-effort heuristics)
_MITRE_HINTS: dict[str, str] = {
    "DDoS": "TA0040 — Impact; T1498 Network Denial of Service",
    "DoS": "TA0040 — Impact; T1499 Endpoint Denial of Service",
    "PortScan": "TA0007 — Discovery; T1046 Network Service Discovery",
    "FTP-Patator": "TA0006 — Credential Access; T1110 Brute Force",
    "SSH-Patator": "TA0006 — Credential Access; T1110.003 Password Spraying",
    "Bot": "TA0011 — Command and Control; T1071 Application Layer Protocol",
    "Web Attack": "TA0001 — Initial Access; T1190 Exploit Public-Facing Application",
    "Infiltration": "TA0003 — Persistence; T1078 Valid Accounts",
    "Heartbleed": "TA0006 — Credential Access; T1212 Exploitation for Credential Access",
}


def _get_mitre_hint(label_text: str) -> Optional[str]:
    for keyword, hint in _MITRE_HINTS.items():
        if keyword.lower() in label_text.lower():
            return hint
    return None


# ---------------------------------------------------------------------------
# AlertCard dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlertCard:
    alert_id: str
    timestamp: str
    severity: str                # "HIGH" | "MEDIUM" | "LOW" | "INFO"
    label_text: str              # "ATTACK" | "BENIGN"
    confidence: float
    user_explanation: str        # NLG-generated adaptive explanation
    top_features: list[dict]     # [{feature, value, shap_value, direction}, ...]
    remediation_steps: list[str]
    mitre_hint: Optional[str]
    raw_flow: dict               # original flow feature dict
    literacy_level: str

    def as_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "label": self.label_text,
            "confidence": round(self.confidence, 4),
            "literacy_level": self.literacy_level,
            "explanation": self.user_explanation,
            "top_features": self.top_features,
            "remediation_steps": self.remediation_steps,
            "mitre_hint": self.mitre_hint,
        }

    def markdown_summary(self) -> str:
        """Render a human-readable Markdown summary of this alert."""
        severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "ℹ️"}.get(
            self.severity, "⚪"
        )
        lines = [
            f"## {severity_emoji} Alert — {self.label_text}",
            f"**ID:** `{self.alert_id}` | **Time:** {self.timestamp} | "
            f"**Confidence:** {self.confidence * 100:.1f}%",
            "",
            "### Explanation",
            self.user_explanation,
            "",
            "### Top Contributing Features",
        ]
        for f in self.top_features[:5]:
            arrow = "▲" if f["direction"] == "increases_risk" else "▼"
            lines.append(
                f"- **{f['feature']}**: {f['value']:.2f}  "
                f"(SHAP {f['shap_value']:+.4f} {arrow})"
            )
        lines += ["", "### Recommended Actions"]
        for i, step in enumerate(self.remediation_steps, 1):
            lines.append(f"{i}. {step}")
        if self.mitre_hint:
            lines += ["", f"**MITRE ATT&CK:** {self.mitre_hint}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RemediationCardBuilder
# ---------------------------------------------------------------------------

class RemediationCardBuilder:
    """Assembles AlertCard objects from detection + explanation + NLG outputs."""

    _counter: int = 0

    def build(
        self,
        explanation: ExplanationResult,
        profile: UserProfile,
        nlg_text: str,
        raw_flow: Optional[dict] = None,
        alert_id: Optional[str] = None,
    ) -> AlertCard:
        """Build a complete AlertCard.

        Args:
            explanation: SHAP ExplanationResult.
            profile: UserProfile for the recipient.
            nlg_text: Pre-generated NLG explanation string.
            raw_flow: Original feature dictionary for the network flow.
            alert_id: Optional explicit ID; auto-generated if not provided.

        Returns:
            Fully populated AlertCard.
        """
        RemediationCardBuilder._counter += 1
        aid = alert_id or f"ALERT-{RemediationCardBuilder._counter:05d}"
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        label_text = "ATTACK" if explanation.predicted_label == 1 else "BENIGN"
        severity = self._compute_severity(explanation)
        remediation = _REMEDIATION_ACTIONS[profile.level][label_text]
        mitre_hint = _get_mitre_hint(label_text) if explanation.predicted_label == 1 else None

        return AlertCard(
            alert_id=aid,
            timestamp=timestamp,
            severity=severity,
            label_text=label_text,
            confidence=explanation.confidence,
            user_explanation=nlg_text,
            top_features=[f.as_dict() for f in explanation.top_features[:10]],
            remediation_steps=remediation,
            mitre_hint=mitre_hint,
            raw_flow=raw_flow or {},
            literacy_level=profile.level.value,
        )

    @staticmethod
    def _compute_severity(explanation: ExplanationResult) -> str:
        if explanation.predicted_label == 0:
            return "INFO"
        confidence = explanation.confidence
        if confidence >= 0.85:
            return "HIGH"
        elif confidence >= 0.60:
            return "MEDIUM"
        else:
            return "LOW"
