# XAI Cybersecurity Alert System

**Reading-level-adaptive, privacy-preserving threat explanation using Random Forest + SHAP + Local LLM**

> ⚠️ Patent-pending research. All rights reserved.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://xai-cybersecurity-alert-system-flmwn94cvuxsyllsvudhsl.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=for-the-badge)]()

🚀 **[Try the live demo →](https://xai-cybersecurity-alert-system-flmwn94cvuxsyllsvudhsl.streamlit.app/)**

---

## Overview

The XAI Cybersecurity Alert System detects network intrusions using a Random Forest classifier trained on CICIDS2017 network flow features, generates per-prediction SHAP explanations, and translates those explanations into plain-language alerts — calibrated to **10 different user personas** spanning complete beginners to security professionals. All processing happens entirely on-device with no data sent to external services.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  XAI Cybersecurity Alert System                   │
│                                                                    │
│   Network Flow (CICIDS2017)                                        │
│         │                                                          │
│         ▼                                                          │
│   ┌─────────────────┐      ┌──────────────────────┐               │
│   │  ThreatDetector │ ───▶ │    XAI Explainer     │               │
│   │  (Random Forest)│      │  (SHAP TreeExplainer)│               │
│   └─────────────────┘      └──────────────────────┘               │
│                                        │                           │
│                          ┌─────────────▼─────────────┐            │
│                          │       UserProfiler         │            │
│                          │  10 Personas × 3 Levels    │            │
│                          │  HOME │ SMB │ ADMIN        │            │
│                          └─────────────┬─────────────┘            │
│                                        │                           │
│                          ┌─────────────▼─────────────┐            │
│                          │       NLG Module           │            │
│                          │  Ollama LLM (on-device)    │            │
│                          │  + Template fallback       │            │
│                          └─────────────┬─────────────┘            │
│                                        │                           │
│                          ┌─────────────▼─────────────┐            │
│                          │   RemediationCardBuilder   │            │
│                          │  AlertCard + MITRE ATT&CK  │            │
│                          └─────────────┬─────────────┘            │
│                                        │                           │
│                          ┌─────────────▼─────────────┐            │
│                          │    Streamlit Dashboard     │            │
│                          │  6 Scenarios │ 5 Tabs      │            │
│                          └───────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Modules

| Module | Description |
|--------|-------------|
| `modules/threat_detector.py` | Random Forest classifier on 60 CICIDS2017 features; synthetic data fallback if no CSV provided |
| `modules/xai_explainer.py` | SHAP `TreeExplainer` — per-prediction feature attribution with direction and magnitude |
| `modules/user_profiler.py` | **10-persona** literacy classifier mapping to 3 reading levels (HOME / SMB / ADMIN) |
| `modules/nlg_module.py` | Persona-aware NLG via local Ollama LLM; template fallback for offline use |
| `modules/remediation_card.py` | Structured `AlertCard` with persona-specific steps and MITRE ATT&CK mapping |
| `app.py` | Streamlit dashboard — entry point for local and cloud deployment |
| `train.py` | CLI training script supporting real CICIDS2017 CSV or synthetic data |

---

## 10 User Personas

One of the core innovations of this system is **persona-adaptive explanations** — the same threat is explained differently depending on who is reading the alert.

| # | Persona | Level | Explanation Style |
|---|---------|-------|-------------------|
| 1 | 👦 Kid | HOME | Super simple words, everyday toy/door analogies, under 50 words |
| 2 | 🧑 Teenager | HOME | Casual language, gaming/social media analogies, practical steps |
| 3 | 🏠 Housewife | HOME | Home and family safety analogies, no jargon |
| 4 | 🛒 Cashier | HOME | One clear action, workplace focus, call manager |
| 5 | 💼 General Employee | HOME | Plain office language, contact IT helpdesk |
| 6 | 🏢 Business Owner | SMB | Business risk, cost and operations impact |
| 7 | 🎓 Student | SMB | Technical context with learning tips, SHAP feature explanation |
| 8 | 👔 Executive / Manager | SMB | One-sentence risk + one-sentence action, no jargon |
| 9 | 📋 Compliance / Auditor | ADMIN | GDPR / NIST / ISO 27001 framing, audit trail steps |
| 10 | 🔐 Security Analyst | ADMIN | Full SHAP detail, MITRE ATT&CK, IR playbook |

---

## 6 Pre-Built Attack Scenarios

| # | Scenario | Attack Type | Key Signals |
|---|----------|-------------|-------------|
| 1 | 💥 DDoS Attack | Volumetric flood | Extremely high packet rate, low IAT, SYN floods |
| 2 | 🔍 Port Scan | Network discovery | Many short flows, single packets, RST flags |
| 3 | ✅ Normal HTTPS Traffic | Benign baseline | Moderate flow, balanced packets, ACK-heavy |
| 4 | 🔑 SSH Brute Force | Credential attack | High SYN count, repeated short flows, fixed port |
| 5 | 🦠 Malware C2 Beacon | Command & Control | Regular heartbeat timing, low byte rate, long duration |
| 6 | 🔗 Suspicious Link / Redirect | Initial access | Short burst, large response, RST after redirect |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Start Ollama for AI-generated explanations

```bash
ollama pull mistral
ollama serve
```

If Ollama is not running, the system automatically falls back to built-in template explanations for all 10 personas.

### 3. Train the model

```bash
# Uses synthetic CICIDS2017-schema data (no CSV needed):
python train.py

# With a real CICIDS2017 CSV:
python train.py --data data/Friday-WorkingHours.csv
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset

Designed for the **CICIDS2017** (Canadian Institute for Cybersecurity Intrusion Detection System 2017) dataset — 60 network flow features covering DDoS, DoS, PortScan, Brute Force, Web Attack, Bot, and Infiltration traffic.

Place any CICIDS2017 CSV file in `data/` and run `train.py`. If no file is provided, the system generates synthetic data with identical feature schema for development and demonstration.

---

## Privacy Design

| Component | Privacy Guarantee |
|-----------|------------------|
| Threat detection | Runs fully local — no traffic data leaves the machine |
| SHAP attribution | Computed in-process — no feature values transmitted |
| LLM explanation | Ollama runs on-device — no prompts sent to cloud APIs |
| Dashboard | No telemetry, no user tracking, no data collection |

---

## Research Contributions

### 1. Persona-Adaptive XAI for Cybersecurity
The first threat explanation system to serve **10 distinct user personas** — from children to compliance auditors — using a unified SHAP-to-NLG pipeline. Each persona receives tailored language, analogies, and detail depth rather than a one-size-fits-all alert.

### 2. Privacy-Preserving Explanation Pipeline
End-to-end on-device processing via **Ollama** eliminates dependency on cloud LLM APIs. Sensitive network telemetry (flow features, SHAP values, IP addresses) never leaves the local environment — a critical requirement for enterprise and government deployments.

### 3. SHAP-to-Remediation Mapping
SHAP feature attributions are automatically translated into **persona-specific, actionable remediation steps** — with MITRE ATT&CK tactic references for security professionals and plain-language instructions for non-technical users.

### 4. Breadth-First Attack Coverage
Six pre-built scenarios covering the full attack lifecycle — from initial access (suspicious links, port scans) through credential attacks (SSH brute force) to impact (DDoS) and persistence (malware C2 beaconing) — enabling demonstrations across the entire MITRE ATT&CK framework.

---

## Citation

```bibtex
@software{irin2024xai,
  author    = {Irin},
  title     = {XAI Cybersecurity Alert System: Reading-Level-Adaptive,
               Privacy-Preserving Threat Explanation using
               Random Forest, SHAP, and Local LLM},
  year      = {2024},
  note      = {Patent Pending},
  url       = {https://github.com/Irin-Akther/xai-cybersecurity-alert-system}
}
```

---

*Developed by Irin — Cybersecurity Research*
