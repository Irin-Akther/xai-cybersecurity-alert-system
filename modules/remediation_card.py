"""
Remediation Card — structured alert card formatter.

Combines threat detection, SHAP explanation, persona-adaptive NLG text, and
actionable remediation guidance into a single structured AlertCard object.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

from modules.user_profiler import LiteracyLevel, Persona, UserProfile
from modules.xai_explainer import ExplanationResult

# ---------------------------------------------------------------------------
# Remediation action library keyed by persona
# ---------------------------------------------------------------------------

_REMEDIATION_ACTIONS: dict[Persona, dict[str, list[str]]] = {
    Persona.KID: {
        "ATTACK": [
            "Tell a grown-up (parent or teacher) right away.",
            "Don't click on anything you don't recognise.",
            "Stop using the internet until an adult checks it.",
        ],
        "BENIGN": ["Everything is fine! Keep using the internet safely."],
    },
    Persona.TEENAGER: {
        "ATTACK": [
            "Disconnect your device from Wi-Fi immediately.",
            "Don't log into any accounts until it's sorted.",
            "Tell a parent, teacher, or IT person what happened.",
            "Run a virus scan on your device.",
        ],
        "BENIGN": ["All clear — no action needed."],
    },
    Persona.HOUSEWIFE: {
        "ATTACK": [
            "Unplug your router for 30 seconds and plug it back in.",
            "Do not enter passwords or card details until it's safe.",
            "Call your internet provider if it keeps happening.",
            "Ask a family member who knows technology to check the device.",
        ],
        "BENIGN": ["Your home internet looks safe. No action needed."],
    },
    Persona.CASHIER: {
        "ATTACK": [
            "Stop using the payment or work system immediately.",
            "Call your manager or supervisor right now.",
            "Do not process any transactions until IT says it's safe.",
            "Write down the time this happened for your records.",
        ],
        "BENIGN": ["System is normal. Continue with your work."],
    },
    Persona.GENERAL_EMPLOYEE: {
        "ATTACK": [
            "Stop using your work device and disconnect from the network.",
            "Contact your IT helpdesk or support team immediately.",
            "Do not open emails or attachments until cleared.",
            "Report the incident using your company's security procedure.",
        ],
        "BENIGN": ["No issues detected. Continue working normally."],
    },
    Persona.BUSINESS_OWNER: {
        "ATTACK": [
            "Isolate the affected device or system from your business network.",
            "Contact your IT provider or managed security service immediately.",
            "Assess whether any customer or financial data may be affected.",
            "Consider notifying customers if data exposure is possible.",
            "Document the incident for insurance and legal purposes.",
        ],
        "BENIGN": ["No threat detected. Normal business operations can continue."],
    },
    Persona.STUDENT: {
        "ATTACK": [
            "Isolate the affected host to prevent lateral spread.",
            "Analyse the flagged flow features in a packet capture tool (Wireshark).",
            "Cross-reference the attack pattern with MITRE ATT&CK framework.",
            "Document findings for your coursework or research notes.",
            "Apply patches and update firewall rules as part of your response exercise.",
        ],
        "BENIGN": [
            "Traffic classified as benign. Good baseline for comparison.",
            "Consider logging this sample to build your normal-traffic dataset.",
        ],
    },
    Persona.EXECUTIVE: {
        "ATTACK": [
            "Authorise IT team to isolate affected systems immediately.",
            "Request a brief incident report within 1 hour.",
            "Assess business continuity impact and notify board if significant.",
            "Engage legal team if customer data may be involved.",
        ],
        "BENIGN": ["No action required. Systems operating normally."],
    },
    Persona.COMPLIANCE: {
        "ATTACK": [
            "Log the incident immediately with full timestamp and evidence.",
            "Assess scope: determine if personal data (PII) was potentially accessed.",
            "If PII involved, initiate GDPR/HIPAA breach notification process within 72 hours.",
            "Document in your ISMS incident register (ISO 27001 A.16).",
            "Notify relevant authorities per NIST IR 800-61 if applicable.",
            "Preserve all logs and artefacts for audit evidence.",
        ],
        "BENIGN": [
            "No incident detected. Log for audit completeness.",
            "Retain this record per your data retention policy.",
        ],
    },
    Persona.SECURITY_ANALYST: {
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

# MITRE ATT&CK tactic hints per attack keyword
_MITRE_HINTS: dict[str, str] = {
    "DDoS":        "TA0040 — Impact; T1498 Network Denial of Service",
    "DoS":         "TA0040 — Impact; T1499 Endpoint Denial of Service",
    "PortScan":    "TA0007 — Discovery; T1046 Network Service Discovery",
    "FTP-Patator": "TA0006 — Credential Access; T1110 Brute Force",
    "SSH-Patator": "TA0006 — Credential Access; T1110.003 Password Spraying",
    "Bot":         "TA0011 — Command and Control; T1071 Application Layer Protocol",
    "Web Attack":  "TA0001 — Initial Access; T1190 Exploit Public-Facing Application",
    "Infiltration":"TA0003 — Persistence; T1078 Valid Accounts",
    "Heartbleed":  "TA0006 — Credential Access; T1212 Exploitation for Credential Access",
    "Malware":     "TA0011 — Command and Control; T1071.001 Web Protocols",
    "Ransomware":  "TA0040 — Impact; T1486 Data Encrypted for Impact",
    "DNS":         "TA0010 — Exfiltration; T1048.003 Exfiltration Over Unencrypted Protocol",
    "Insider":     "TA0009 — Collection; T1078 Valid Accounts",
    "SQL":         "TA0001 — Initial Access; T1190 Exploit Public-Facing Application",
    "Exfil":       "TA0010 — Exfiltration; T1048 Exfiltration Over Alternative Protocol",
    "Suspicious":  "TA0001 — Initial Access; T1566 Phishing",
}


def _get_mitre_hint(label_text: str) -> Optional[str]:
    for keyword, hint in _MITRE_HINTS.items():
        if keyword.lower() in label_text.lower():
            return hint
    return None


# ---------------------------------------------------------------------------
# AlertCard
# ---------------------------------------------------------------------------

def _confidence_explanation(confidence: float) -> str:
    """Plain-language description of model confidence for non-technical users."""
    if confidence >= 0.85:
        return (
            "The system is highly confident in this detection. "
            "The network pattern strongly matches known attack signatures."
        )
    if confidence >= 0.60:
        return (
            "The system is moderately confident. "
            "This pattern is suspicious but could have other explanations — treat it as a warning."
        )
    if confidence >= 0.40:
        return (
            "The system has low confidence. "
            "This may be a false alarm, but it is worth a quick check."
        )
    return (
        "The system flagged this for informational purposes only. "
        "No immediate action is needed."
    )


@dataclass
class AlertCard:
    alert_id: str
    timestamp: str
    severity: str
    label_text: str
    confidence: float
    confidence_explanation: str
    user_explanation: str
    top_features: list[dict]
    remediation_steps: list[str]
    mitre_hint: Optional[str]
    raw_flow: dict
    literacy_level: str
    persona: str

    def as_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "label": self.label_text,
            "confidence": round(self.confidence, 4),
            "confidence_explanation": self.confidence_explanation,
            "literacy_level": self.literacy_level,
            "persona": self.persona,
            "explanation": self.user_explanation,
            "top_features": self.top_features,
            "remediation_steps": self.remediation_steps,
            "mitre_hint": self.mitre_hint,
        }

    def markdown_summary(self) -> str:
        severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "ℹ️"}.get(self.severity, "⚪")
        lines = [
            f"## {severity_emoji} Alert — {self.label_text}",
            f"**ID:** `{self.alert_id}` | **Time:** {self.timestamp} | "
            f"**Confidence:** {self.confidence * 100:.1f}% | **Profile:** {self.persona}",
            "",
            "### Explanation",
            self.user_explanation,
            "",
            "### Top Contributing Features",
        ]
        for f in self.top_features[:5]:
            arrow = "▲" if f["direction"] == "increases_risk" else "▼"
            lines.append(f"- **{f['feature']}**: {f['value']:.2f}  (SHAP {f['shap_value']:+.4f} {arrow})")
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
        RemediationCardBuilder._counter += 1
        aid = alert_id or f"ALERT-{RemediationCardBuilder._counter:05d}"
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        label_text = "ATTACK" if explanation.predicted_label == 1 else "BENIGN"
        severity = self._compute_severity(explanation)
        remediation = _REMEDIATION_ACTIONS[profile.persona][label_text]
        mitre_hint = _get_mitre_hint(label_text) if explanation.predicted_label == 1 else None

        return AlertCard(
            alert_id=aid,
            timestamp=timestamp,
            severity=severity,
            label_text=label_text,
            confidence=explanation.confidence,
            confidence_explanation=_confidence_explanation(explanation.confidence),
            user_explanation=nlg_text,
            top_features=[f.as_dict() for f in explanation.top_features[:10]],
            remediation_steps=remediation,
            mitre_hint=mitre_hint,
            raw_flow=raw_flow or {},
            literacy_level=profile.level.value,
            persona=profile.persona.value,
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
