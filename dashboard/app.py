"""
XAI Cybersecurity Alert System — Streamlit Dashboard

Reading-level-adaptive, privacy-preserving threat explanation interface.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on the path when running via `streamlit run dashboard/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from modules.nlg_module import NLGModule
from modules.remediation_card import AlertCard, RemediationCardBuilder
from modules.threat_detector import FEATURE_COLS, ThreatDetector
from modules.user_profiler import (
    ADMIN_PROFILE,
    HOME_PROFILE,
    SMB_PROFILE,
    LiteracyLevel,
    UserProfiler,
    UserProfile,
)
from modules.xai_explainer import XAIExplainer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="XAI Cybersecurity Alert System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    if "detector" not in st.session_state:
        st.session_state.detector = None
    if "explainer" not in st.session_state:
        st.session_state.explainer = None
    if "nlg" not in st.session_state:
        st.session_state.nlg = NLGModule()
    if "builder" not in st.session_state:
        st.session_state.builder = RemediationCardBuilder()
    if "profiler" not in st.session_state:
        st.session_state.profiler = UserProfiler()
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = HOME_PROFILE
    if "alert_history" not in st.session_state:
        st.session_state.alert_history: list[AlertCard] = []
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "models"


@st.cache_resource(show_spinner="Loading threat detection model…")
def load_or_train_detector() -> ThreatDetector:
    model_path = MODEL_DIR / "random_forest.joblib"
    if model_path.exists():
        return ThreatDetector.load(MODEL_DIR)
    detector = ThreatDetector()
    detector.train()
    detector.save(MODEL_DIR)
    return detector


def get_explainer(detector: ThreatDetector) -> XAIExplainer:
    return XAIExplainer(detector, top_n=10)


def severity_color(severity: str) -> str:
    return {
        "HIGH": "#ff4b4b",
        "MEDIUM": "#ffa500",
        "LOW": "#00c853",
        "INFO": "#1e88e5",
    }.get(severity, "#888")


def _render_feature_bar(top_features: list[dict]):
    """Render a simple horizontal bar chart of SHAP values."""
    names = [f["feature"].replace("_", " ")[:30] for f in top_features]
    shap_vals = [f["shap_value"] for f in top_features]
    colors = ["#ff4b4b" if v > 0 else "#1e88e5" for v in shap_vals]
    chart_data = pd.DataFrame({"feature": names, "SHAP value": shap_vals})
    chart_data = chart_data.sort_values("SHAP value")
    st.bar_chart(chart_data.set_index("feature"))


# ---------------------------------------------------------------------------
# Sidebar — Model & Profile
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/cyber-security.png", width=80)
    st.title("XAI Cybersecurity\nAlert System")
    st.caption("v1.0 | Privacy-Preserving | Patent-Pending")
    st.divider()

    # --- Model section ---
    st.subheader("🤖 Detection Model")
    if st.button("Load / Train Model", use_container_width=True):
        with st.spinner("Training Random Forest on CICIDS2017 data…"):
            detector = load_or_train_detector()
            st.session_state.detector = detector
            st.session_state.explainer = get_explainer(detector)
            st.session_state.model_trained = True
        st.success("Model ready!")

    if st.session_state.model_trained:
        st.success("✅ Model loaded")
    else:
        st.warning("⚠️ Model not loaded. Click above.")

    st.divider()

    # --- User Profile section ---
    st.subheader("👤 User Profile")
    profile_option = st.selectbox(
        "Select your expertise level",
        options=["HOME — Home User", "SMB — IT Staff", "ADMIN — Security Analyst"],
        index=0,
    )
    level_map = {
        "HOME — Home User": HOME_PROFILE,
        "SMB — IT Staff": SMB_PROFILE,
        "ADMIN — Security Analyst": ADMIN_PROFILE,
    }
    st.session_state.user_profile = level_map[profile_option]
    profile: UserProfile = st.session_state.user_profile
    st.info(f"Reading level: **{profile.reading_grade}**")

    st.divider()

    # --- Ollama status ---
    st.subheader("🔒 Local LLM (Ollama)")
    nlg: NLGModule = st.session_state.nlg
    available_models = nlg.list_available_models()
    if available_models:
        chosen_model = st.selectbox("Select model", available_models, index=0)
        nlg.set_model(chosen_model.split(":")[0])
        st.success(f"Ollama: ✅ {chosen_model}")
    else:
        st.warning("Ollama not running — template explanations will be used.")
        st.caption("Start Ollama and run: `ollama pull mistral`")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_detect, tab_batch, tab_history, tab_questionnaire, tab_about = st.tabs(
    ["🔍 Detect Threat", "📂 Batch Analysis", "📋 Alert History", "📝 Profile Quiz", "ℹ️ About"]
)

# ===========================================================================
# TAB 1 — Single Flow Detection
# ===========================================================================
with tab_detect:
    st.header("Single Network Flow Analysis")
    st.markdown(
        "Enter network flow features to classify the traffic and receive an "
        "Explainable AI explanation tailored to your expertise level."
    )

    col_left, col_right = st.columns([1, 1])

    # Preset scenarios
    with col_left:
        st.subheader("Quick Scenarios")
        scenario = st.selectbox(
            "Load a pre-built scenario",
            [
                "— Manual entry —",
                "DDoS Attack (high packet rate)",
                "Port Scan (many short flows)",
                "Normal HTTPS traffic",
                "SSH Brute Force attempt",
            ],
        )

    SCENARIO_DATA: dict[str, dict] = {
        "DDoS Attack (high packet rate)": {
            "Flow Duration": 1500,
            "Total Fwd Packets": 5000,
            "Total Backward Packets": 100,
            "Flow Bytes/s": 8_000_000,
            "Flow Packets/s": 500_000,
            "SYN Flag Count": 5,
            "RST Flag Count": 3,
            "Flow IAT Mean": 50,
            "Fwd Packet Length Mean": 64.0,
            "Bwd Packet Length Mean": 40.0,
        },
        "Port Scan (many short flows)": {
            "Flow Duration": 200,
            "Total Fwd Packets": 1,
            "Total Backward Packets": 0,
            "Flow Bytes/s": 500,
            "Flow Packets/s": 5000,
            "SYN Flag Count": 1,
            "RST Flag Count": 1,
            "Flow IAT Mean": 100,
            "Fwd Packet Length Mean": 40.0,
        },
        "Normal HTTPS traffic": {
            "Flow Duration": 800_000,
            "Total Fwd Packets": 12,
            "Total Backward Packets": 10,
            "Flow Bytes/s": 12_000,
            "Flow Packets/s": 18,
            "ACK Flag Count": 8,
            "Fwd Packet Length Mean": 600.0,
            "Bwd Packet Length Mean": 500.0,
            "Flow IAT Mean": 50_000,
        },
        "SSH Brute Force attempt": {
            "Flow Duration": 200_000,
            "Total Fwd Packets": 300,
            "Total Backward Packets": 200,
            "Flow Bytes/s": 2_500,
            "Flow Packets/s": 200,
            "SYN Flag Count": 50,
            "FIN Flag Count": 10,
            "Fwd Packet Length Mean": 32.0,
            "Flow IAT Mean": 600,
        },
    }

    prefill = SCENARIO_DATA.get(scenario, {})

    st.subheader("Flow Features")
    feature_cols_1 = st.columns(3)
    user_features: dict = {}

    # We show only the most interpretable features in the UI; others default to 0
    ui_features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Flow Bytes/s", "Flow Packets/s",
        "Fwd Packet Length Mean", "Bwd Packet Length Mean",
        "Flow IAT Mean", "SYN Flag Count", "RST Flag Count",
        "ACK Flag Count", "FIN Flag Count",
    ]
    for idx, feat in enumerate(ui_features):
        col = feature_cols_1[idx % 3]
        default_val = float(prefill.get(feat, 0.0))
        user_features[feat] = col.number_input(
            feat, value=default_val, format="%.2f", key=f"feat_{feat}"
        )

    if st.button("🔍 Analyse Flow", use_container_width=True, type="primary"):
        if not st.session_state.model_trained:
            st.error("Please load the model first (sidebar).")
        else:
            detector: ThreatDetector = st.session_state.detector
            explainer: XAIExplainer = st.session_state.explainer
            builder: RemediationCardBuilder = st.session_state.builder
            nlg: NLGModule = st.session_state.nlg
            profile: UserProfile = st.session_state.user_profile

            with st.spinner("Analysing…"):
                explanation = explainer.explain_single(user_features)
                nlg_text = nlg.generate(explanation, profile)
                card = builder.build(explanation, profile, nlg_text, raw_flow=user_features)
                st.session_state.alert_history.insert(0, card)

            sev_color = severity_color(card.severity)
            st.markdown(
                f"<div style='background:{sev_color}20;border-left:5px solid {sev_color};"
                f"padding:12px;border-radius:4px;'>"
                f"<h3 style='color:{sev_color};margin:0'>{card.severity} — {card.label_text}</h3>"
                f"<p style='margin:4px 0'>Confidence: {card.confidence*100:.1f}% | "
                f"Alert ID: {card.alert_id}</p></div>",
                unsafe_allow_html=True,
            )
            st.markdown("### Explanation")
            st.write(card.user_explanation)

            with st.expander("📊 SHAP Feature Contributions", expanded=True):
                _render_feature_bar(card.top_features)

            with st.expander("🛠️ Remediation Steps"):
                for i, step in enumerate(card.remediation_steps, 1):
                    st.write(f"{i}. {step}")
                if card.mitre_hint:
                    st.info(f"**MITRE ATT&CK:** {card.mitre_hint}")

            with st.expander("📄 Full Alert Card (JSON)"):
                st.json(card.as_dict())

# ===========================================================================
# TAB 2 — Batch CSV Analysis
# ===========================================================================
with tab_batch:
    st.header("Batch Network Flow Analysis")
    st.markdown("Upload a CSV file with CICIDS2017-format columns to analyse multiple flows.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        if not st.session_state.model_trained:
            st.error("Please load the model first (sidebar).")
        else:
            df_upload = pd.read_csv(uploaded, low_memory=False)
            df_upload.columns = df_upload.columns.str.strip()
            st.info(f"Loaded {len(df_upload):,} rows, {len(df_upload.columns)} columns.")

            if st.button("Run Batch Detection", type="primary"):
                detector = st.session_state.detector
                progress = st.progress(0, text="Running detection…")

                X = df_upload.copy()
                for col in FEATURE_COLS:
                    if col not in X.columns:
                        X[col] = 0.0

                preds = detector.predict(X)
                probas = detector.predict_proba(X)

                results_df = df_upload.copy()
                results_df["Prediction"] = ["ATTACK" if p == 1 else "BENIGN" for p in preds]
                results_df["Attack_Probability"] = probas[:, 1].round(4)
                progress.progress(100, text="Done!")

                st.subheader("Summary")
                attack_count = int(preds.sum())
                benign_count = int(len(preds) - attack_count)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Flows", len(preds))
                c2.metric("Attacks Detected", attack_count, delta=None)
                c3.metric("Benign Flows", benign_count)

                st.subheader("Results")
                st.dataframe(
                    results_df[["Prediction", "Attack_Probability"] + list(df_upload.columns[:5])],
                    use_container_width=True,
                )

                csv_out = results_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv_out,
                    "xai_detection_results.csv",
                    "text/csv",
                )

# ===========================================================================
# TAB 3 — Alert History
# ===========================================================================
with tab_history:
    st.header("Alert History")
    history: list[AlertCard] = st.session_state.alert_history

    if not history:
        st.info("No alerts yet. Run a detection in the 'Detect Threat' tab.")
    else:
        st.metric("Total Alerts This Session", len(history))
        for card in history:
            sev_color = severity_color(card.severity)
            with st.expander(
                f"{card.severity} | {card.label_text} | {card.confidence*100:.1f}% | {card.timestamp}"
            ):
                st.markdown(
                    f"<span style='color:{sev_color};font-weight:bold'>{card.severity}</span> — "
                    f"**{card.label_text}** | Literacy: {card.literacy_level}",
                    unsafe_allow_html=True,
                )
                st.write(card.user_explanation)
                if card.top_features:
                    _render_feature_bar(card.top_features[:5])

        if st.button("🗑️ Clear History"):
            st.session_state.alert_history = []
            st.rerun()

# ===========================================================================
# TAB 4 — User Profile Questionnaire
# ===========================================================================
with tab_questionnaire:
    st.header("Cybersecurity Literacy Quiz")
    st.markdown(
        "Take this short quiz to automatically calibrate alert explanations "
        "to your expertise level."
    )

    from modules.user_profiler import QUESTIONS

    answers: dict[str, str] = {}
    for q in QUESTIONS:
        st.markdown(f"**{q['text']}**")
        options_labels = [f"[{k}] {label}" for k, (label, _) in q["options"].items()]
        choice_label = st.radio(
            q["text"],
            options_labels,
            label_visibility="collapsed",
            key=f"quiz_{q['id']}",
        )
        chosen_key = choice_label.split("]")[0].strip("[")
        answers[q["id"]] = chosen_key

    user_name = st.text_input("Your name or alias (optional)")
    if st.button("Submit Quiz & Update Profile", type="primary"):
        profiler: UserProfiler = st.session_state.profiler
        new_profile = profiler.from_answers(answers, display_name=user_name or "User")
        st.session_state.user_profile = new_profile
        level_labels = {
            LiteracyLevel.HOME: ("HOME — Home User", "You'll receive simple, jargon-free explanations."),
            LiteracyLevel.SMB: ("SMB — IT Staff", "You'll receive technical but accessible explanations."),
            LiteracyLevel.ADMIN: ("ADMIN — Security Analyst", "You'll receive full forensic-level detail."),
        }
        label, desc = level_labels[new_profile.level]
        st.success(f"Profile set to **{label}**. {desc}")

# ===========================================================================
# TAB 5 — About
# ===========================================================================
with tab_about:
    st.header("About This System")
    st.markdown("""
## XAI Cybersecurity Alert System

**Version:** 1.0
**Research Status:** Patent-Pending

### System Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Threat Detector | Random Forest (scikit-learn) | Network intrusion classification on CICIDS2017 features |
| XAI Explainer | SHAP TreeExplainer | Per-prediction feature attribution and interpretability |
| User Profiler | Rule-based + questionnaire | 3-level literacy classification (HOME / SMB / ADMIN) |
| NLG Module | Ollama (Mistral / LLaMA 3) | Privacy-preserving, on-device natural language generation |
| Remediation Engine | Curated action library | Level-appropriate remediation guidance |
| Dashboard | Streamlit | Interactive web interface |

### Privacy-Preserving Design

All inference — both threat detection and explanation generation — runs entirely
on the local machine. No network traffic data, feature values, or SHAP outputs
are transmitted to external services.

### Dataset

Trained on the **CICIDS2017** (Canadian Institute for Cybersecurity Intrusion
Detection System) dataset, which provides labelled network flow records for
multiple attack categories including DDoS, DoS, PortScan, Brute Force, and Web Attacks.

### Research Contributions

1. **Reading-level adaptive XAI:** SHAP explanations are translated into
   plain language at three calibrated literacy levels using a local LLM.

2. **Privacy-first explanation pipeline:** End-to-end on-device processing
   eliminates the need to share sensitive network telemetry with cloud APIs.

3. **Structured remediation integration:** SHAP feature attribution is
   automatically mapped to actionable remediation steps and MITRE ATT&CK tactics.

---
*Developed by Irin — Cybersecurity Research | NIW Research Portfolio*
    """)
