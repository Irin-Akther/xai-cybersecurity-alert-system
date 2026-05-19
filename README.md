# XAI Cybersecurity Alert System

**Reading-level-adaptive, privacy-preserving threat explanation using Random Forest + SHAP + Local LLM**

> ⚠️ Patent-pending research. All rights reserved.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://xai-cybersecurity-alert-system-flmwn94cvuxsyllsvudhsl.streamlit.app/)

🚀 **[Try the live demo →](https://xai-cybersecurity-alert-system-flmwn94cvuxsyllsvudhsl.streamlit.app/)**

---

## Overview

The XAI Cybersecurity Alert System detects network intrusions using a Random Forest classifier
trained on CICIDS2017 features, generates per-prediction SHAP explanations, and translates
those explanations into plain-language alerts calibrated to the recipient's cybersecurity
literacy level — entirely on-device.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               XAI Cybersecurity Alert System         │
│                                                       │
│  Network Flow  →  ThreatDetector  →  XAIExplainer   │
│  (CICIDS2017)     (Random Forest)     (SHAP)          │
│                         │                │            │
│                   UserProfiler      NLGModule         │
│                   (HOME/SMB/ADMIN)  (Ollama LLM)      │
│                         │                │            │
│                   RemediationCardBuilder              │
│                         │                             │
│                   Streamlit Dashboard                 │
└─────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `modules/threat_detector.py` | Random Forest classifier; auto-trains on synthetic CICIDS2017 data if no CSV is present |
| `modules/xai_explainer.py` | SHAP `TreeExplainer` for per-prediction feature attribution |
| `modules/user_profiler.py` | 3-level literacy classifier: **HOME**, **SMB**, **ADMIN** |
| `modules/nlg_module.py` | Calls local Ollama LLM for privacy-preserving NL explanation; falls back to templates |
| `modules/remediation_card.py` | Assembles structured `AlertCard` objects with remediation steps and MITRE ATT&CK hints |
| `dashboard/app.py` | Streamlit interactive dashboard |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Start Ollama for LLM explanations

```bash
ollama pull mistral
ollama serve
```

### 3. Train the model

```bash
# Without a dataset (uses synthetic CICIDS2017-schema data):
python train.py

# With a real CICIDS2017 CSV:
python train.py --data data/Friday-WorkingHours.csv
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Dataset

This system is designed for the **CICIDS2017** dataset published by the Canadian Institute
for Cybersecurity. Download the CSV files from the official source and place them in `data/`.

If no CSV is provided, the system automatically generates synthetic data with the same
feature schema for development and demonstration purposes.

## Privacy Design

All computation — threat detection, SHAP attribution, and LLM inference — runs locally.
No network traffic data or feature values are transmitted to external APIs.

## Research Contributions

1. **Reading-level adaptive XAI:** First system to combine SHAP explanations with a
   three-tier literacy model (HOME/SMB/ADMIN) for cybersecurity alerts.

2. **Privacy-preserving explanation pipeline:** End-to-end on-device processing via
   Ollama eliminates dependency on cloud LLM APIs.

3. **Structured remediation integration:** SHAP feature attribution is automatically
   mapped to actionable steps and MITRE ATT&CK tactic references.

## Citation

If you use this work in your research, please cite:

```
Irin (2024). XAI Cybersecurity Alert System: Reading-Level-Adaptive,
Privacy-Preserving Threat Explanation using Random Forest, SHAP, and Local LLM.
Patent Pending.
```
