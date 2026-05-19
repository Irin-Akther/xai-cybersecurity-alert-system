"""
XAI Cybersecurity Alert System — Streamlit Dashboard

Reading-level-adaptive, privacy-preserving threat explanation interface.
Supports 10 user personas and 6 pre-built attack scenarios.

Entry point for Streamlit Cloud deployment (main file: app.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Root of the repo — modules/ is a direct sibling of this file
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

import csv
import datetime
import os

from modules.nlg_module import NLGModule, _template_fallback
from modules.remediation_card import AlertCard, RemediationCardBuilder
from modules.threat_detector import FEATURE_COLS, ThreatDetector
from modules.user_profiler import (
    LiteracyLevel,
    Persona,
    UserProfile,
    UserProfiler,
    make_profile,
)
from modules.xai_explainer import ExplanationResult, FeatureContribution, XAIExplainer

# On Streamlit Cloud /mount/src is read-only; write user-study responses to /tmp
_IS_CLOUD = os.path.exists("/mount/src")
_RESPONSES_PATH = (
    Path("/tmp/user_study_responses.csv")
    if _IS_CLOUD
    else PROJECT_ROOT / "data" / "user_study_responses.csv"
)

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
# All personas with display labels and emoji
# ---------------------------------------------------------------------------
PERSONA_OPTIONS: list[tuple[str, Persona]] = [
    ("👦 Kid",                  Persona.KID),
    ("🧑 Teenager",             Persona.TEENAGER),
    ("🏠 Housewife",            Persona.HOUSEWIFE),
    ("🛒 Cashier",              Persona.CASHIER),
    ("💼 General Employee",     Persona.GENERAL_EMPLOYEE),
    ("🏢 Business Owner",       Persona.BUSINESS_OWNER),
    ("🎓 Student",              Persona.STUDENT),
    ("👔 Executive / Manager",  Persona.EXECUTIVE),
    ("📋 Compliance / Auditor", Persona.COMPLIANCE),
    ("🔐 Security Analyst",     Persona.SECURITY_ANALYST),
]
PERSONA_LABEL_MAP = {label: persona for label, persona in PERSONA_OPTIONS}

# ---------------------------------------------------------------------------
# Pre-built scenarios
# ---------------------------------------------------------------------------
SCENARIO_DATA: dict[str, dict] = {
    "DDoS Attack": {
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
    "Port Scan": {
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
    "Normal HTTPS Traffic": {
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
    "SSH Brute Force": {
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
    "Malware C2 Beacon": {
        "Flow Duration": 30_000_000,
        "Total Fwd Packets": 480,
        "Total Backward Packets": 480,
        "Flow Bytes/s": 800,
        "Flow Packets/s": 32,
        "ACK Flag Count": 480,
        "PSH Flag Count": 240,
        "Fwd Packet Length Mean": 128.0,
        "Bwd Packet Length Mean": 256.0,
        "Flow IAT Mean": 60_000,
        "Flow IAT Std": 500.0,
    },
    "Suspicious Link / Redirect": {
        "Flow Duration": 5_000,
        "Total Fwd Packets": 4,
        "Total Backward Packets": 6,
        "Flow Bytes/s": 95_000,
        "Flow Packets/s": 2_000,
        "SYN Flag Count": 1,
        "FIN Flag Count": 1,
        "ACK Flag Count": 4,
        "Fwd Packet Length Mean": 200.0,
        "Bwd Packet Length Mean": 1400.0,
        "Flow IAT Mean": 800,
        "RST Flag Count": 1,
    },
}

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "DDoS Attack":               "Massive packet flood aimed at taking down a server",
    "Port Scan":                 "Attacker probing for open ports on your network",
    "Normal HTTPS Traffic":      "Regular secure web browsing — no threat",
    "SSH Brute Force":           "Repeated login attempts trying to guess your password",
    "Malware C2 Beacon":         "Infected device secretly checking in with attacker server",
    "Suspicious Link / Redirect":"Malicious website redirect attempting drive-by download",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "detector": None,
        "explainer": None,
        "nlg": NLGModule(),
        "builder": RemediationCardBuilder(),
        "profiler": UserProfiler(),
        "user_profile": make_profile(Persona.GENERAL_EMPLOYEE),
        "alert_history": [],
        "model_trained": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

MODEL_DIR = PROJECT_ROOT / "models"


@st.cache_resource(show_spinner="Training threat detection model…")
def load_or_train_detector() -> ThreatDetector:
    model_path = MODEL_DIR / "random_forest.joblib"
    if model_path.exists():
        return ThreatDetector.load(MODEL_DIR)
    detector = ThreatDetector()
    detector.train()
    detector.save(MODEL_DIR)
    return detector


def severity_color(severity: str) -> str:
    return {"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#00c853", "INFO": "#1e88e5"}.get(severity, "#888")


def _render_feature_bar(top_features: list[dict]):
    names = [f["feature"].replace("_", " ")[:30] for f in top_features]
    shap_vals = [f["shap_value"] for f in top_features]
    chart_data = pd.DataFrame({"feature": names, "SHAP value": shap_vals}).sort_values("SHAP value")
    st.bar_chart(chart_data.set_index("feature"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ XAI Cybersecurity\nAlert System")
    st.caption("v2.0 | Privacy-Preserving | Patent-Pending")
    st.divider()

    st.subheader("🤖 Detection Model")
    if st.button("Load / Train Model", use_container_width=True):
        with st.spinner("Loading model…"):
            detector = load_or_train_detector()
            st.session_state.detector = detector
            st.session_state.explainer = XAIExplainer(detector, top_n=10)
            st.session_state.model_trained = True
        st.success("Model ready!")

    if st.session_state.model_trained:
        st.success("✅ Model loaded")
    else:
        st.warning("⚠️ Click above to load model")

    st.divider()

    st.subheader("👤 User Profile")
    persona_labels = [label for label, _ in PERSONA_OPTIONS]
    selected_label = st.selectbox("Who are you?", persona_labels, index=4)
    selected_persona = PERSONA_LABEL_MAP[selected_label]
    profile = make_profile(selected_persona)
    st.session_state.user_profile = profile
    level_badge = {"HOME": "🟢 Simple language", "SMB": "🟡 Business language", "ADMIN": "🔴 Full technical detail"}
    st.info(level_badge.get(profile.level.value, ""))

    st.divider()

    st.subheader("🔒 Local LLM (Ollama)")
    nlg: NLGModule = st.session_state.nlg
    available_models = nlg.list_available_models()
    if available_models:
        chosen = st.selectbox("Model", available_models)
        nlg.set_model(chosen.split(":")[0])
        st.success(f"✅ {chosen}")
    else:
        st.warning("Ollama not running — template mode")
        st.caption("`ollama pull mistral && ollama serve`")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_detect, tab_batch, tab_history, tab_quiz, tab_about, tab_study = st.tabs(
    ["🔍 Detect Threat", "📂 Batch Analysis", "📋 Alert History", "📝 Profile Quiz", "ℹ️ About", "👥 User Study"]
)

# ===========================================================================
# TAB 1 — Single Flow Detection
# ===========================================================================
with tab_detect:
    st.header("Single Network Flow Analysis")

    profile: UserProfile = st.session_state.user_profile
    st.markdown(
        f"Explaining alerts as: **{profile.persona.value}** "
        f"({profile.level.value} level — {profile.reading_grade})"
    )
    st.divider()

    ui_features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Flow Bytes/s", "Flow Packets/s",
        "Fwd Packet Length Mean", "Bwd Packet Length Mean",
        "Flow IAT Mean", "Flow IAT Std",
        "SYN Flag Count", "RST Flag Count", "ACK Flag Count",
        "FIN Flag Count", "PSH Flag Count",
    ]

    st.subheader("Quick Scenarios")
    cols_scenarios = st.columns(3)
    for idx, (name, desc) in enumerate(SCENARIO_DESCRIPTIONS.items()):
        col = cols_scenarios[idx % 3]
        if col.button(f"**{name}**\n\n{desc}", key=f"btn_{name}", use_container_width=True):
            st.session_state["active_scenario"] = name
            # Explicitly write into each widget's session-state key so Streamlit
            # picks up the new value even though the keys already exist.
            scenario_vals = SCENARIO_DATA[name]
            for feat in ui_features:
                st.session_state[f"feat_{feat}"] = float(scenario_vals.get(feat, 0.0))

    active = st.session_state.get("active_scenario", "— Manual entry —")
    if active != "— Manual entry —":
        st.info(f"Loaded scenario: **{active}** — {SCENARIO_DESCRIPTIONS.get(active, '')}")

    st.divider()
    st.subheader("Flow Feature Values")
    user_features: dict = {}
    feat_cols = st.columns(3)
    for idx, feat in enumerate(ui_features):
        col = feat_cols[idx % 3]
        user_features[feat] = col.number_input(feat, format="%.2f", key=f"feat_{feat}")

    if st.button("🔍 Analyse Flow", use_container_width=True, type="primary"):
        if not st.session_state.model_trained:
            st.error("Please load the model first (sidebar).")
        else:
            with st.spinner("Analysing…"):
                explainer: XAIExplainer = st.session_state.explainer
                builder: RemediationCardBuilder = st.session_state.builder
                nlg: NLGModule = st.session_state.nlg
                profile: UserProfile = st.session_state.user_profile

                explanation = explainer.explain_single(user_features)
                nlg_text = nlg.generate(explanation, profile)
                card = builder.build(explanation, profile, nlg_text, raw_flow=user_features)
                st.session_state.alert_history.insert(0, card)

            sev = card.severity
            sev_color = severity_color(sev)
            st.markdown(
                f"<div style='background:{sev_color}20;border-left:5px solid {sev_color};"
                f"padding:14px;border-radius:6px;margin-bottom:12px'>"
                f"<h3 style='color:{sev_color};margin:0'>{sev} — {card.label_text}</h3>"
                f"<p style='margin:4px 0 0'>Confidence: {card.confidence*100:.1f}% | "
                f"Profile: {card.persona} | ID: {card.alert_id}</p></div>",
                unsafe_allow_html=True,
            )

            st.markdown("### What this means for you")
            st.write(card.user_explanation)

            st.markdown("**🎯 How certain is this detection?**")
            st.info(card.confidence_explanation)

            with st.expander("📊 SHAP Feature Contributions", expanded=True):
                st.caption("Features pushing toward ATTACK (red/positive) vs BENIGN (blue/negative)")
                _render_feature_bar(card.top_features)

            with st.expander("🛠️ What you should do"):
                for i, step in enumerate(card.remediation_steps, 1):
                    st.write(f"{i}. {step}")
                if card.mitre_hint:
                    st.info(f"**MITRE ATT&CK:** {card.mitre_hint}")

            with st.expander("📄 Full Alert JSON"):
                st.json(card.as_dict())

# ===========================================================================
# TAB 2 — Batch Analysis
# ===========================================================================
with tab_batch:
    st.header("Batch Network Flow Analysis")
    st.markdown("Upload a CSV with CICIDS2017-format columns to analyse multiple flows at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        if not st.session_state.model_trained:
            st.error("Please load the model first.")
        else:
            df_upload = pd.read_csv(uploaded, low_memory=False)
            df_upload.columns = df_upload.columns.str.strip()
            st.info(f"Loaded {len(df_upload):,} rows, {len(df_upload.columns)} columns.")

            if st.button("Run Batch Detection", type="primary"):
                detector: ThreatDetector = st.session_state.detector
                X = df_upload.copy()
                for col in FEATURE_COLS:
                    if col not in X.columns:
                        X[col] = 0.0

                with st.spinner("Running detection…"):
                    preds = detector.predict(X)
                    probas = detector.predict_proba(X)

                results_df = df_upload.copy()
                results_df["Prediction"] = ["ATTACK" if p == 1 else "BENIGN" for p in preds]
                results_df["Attack_Probability"] = probas[:, 1].round(4)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Flows", len(preds))
                c2.metric("Attacks Detected", int(preds.sum()))
                c3.metric("Benign Flows", int(len(preds) - preds.sum()))

                st.dataframe(
                    results_df[["Prediction", "Attack_Probability"] + list(df_upload.columns[:5])],
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Download Results CSV",
                    results_df.to_csv(index=False).encode(),
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
                f"{card.severity} | {card.label_text} | {card.confidence*100:.1f}% | {card.persona} | {card.timestamp}"
            ):
                st.markdown(
                    f"<span style='color:{sev_color};font-weight:bold'>{card.severity}</span> — "
                    f"**{card.label_text}** | Profile: {card.persona}",
                    unsafe_allow_html=True,
                )
                st.write(card.user_explanation)
                if card.top_features:
                    _render_feature_bar(card.top_features[:5])

        if st.button("🗑️ Clear History"):
            st.session_state.alert_history = []
            st.session_state.pop("active_scenario", None)
            st.rerun()

# ===========================================================================
# TAB 4 — Profile Quiz
# ===========================================================================
with tab_quiz:
    st.header("Find Your Profile")
    st.markdown("Answer 4 quick questions and we'll set your explanation style automatically.")

    from modules.user_profiler import QUESTIONS

    answers: dict[str, str] = {}
    for q in QUESTIONS:
        st.markdown(f"**{q['text']}**")
        options_labels = [f"[{k}] {label}" for k, (label, _) in q["options"].items()]
        choice_label = st.radio(q["text"], options_labels, label_visibility="collapsed", key=f"quiz_{q['id']}")
        answers[q["id"]] = choice_label.split("]")[0].strip("[")

    user_name = st.text_input("Your name or alias (optional)")
    if st.button("Submit & Set My Profile", type="primary"):
        profiler: UserProfiler = st.session_state.profiler
        new_profile = profiler.from_answers(answers, display_name=user_name or "User")
        st.session_state.user_profile = new_profile
        st.success(
            f"Profile set to **{new_profile.persona.value}** "
            f"({new_profile.level.value} level). "
            "Alerts will now be explained in your style."
        )

# ===========================================================================
# TAB 5 — About
# ===========================================================================
# ===========================================================================
# TAB 6 — User Study
# ===========================================================================
with tab_study:
    st.header("User Study — Readability Feedback")
    st.markdown(
        "Help us improve! Read the sample alert below and answer 5 quick questions. "
        "Your responses are stored anonymously and used only for research."
    )
    st.divider()

    # --- Sample alert rendered for General Employee persona ---
    st.subheader("Sample Alert")
    sample_profile = make_profile(Persona.GENERAL_EMPLOYEE)
    _sample_exp = ExplanationResult(
        predicted_label=1,
        confidence=0.91,
        base_value=0.5,
        top_features=[
            FeatureContribution("Flow Packets/s", 450000.0, 0.38, "increases_risk"),
            FeatureContribution("SYN Flag Count", 5.0, 0.21, "increases_risk"),
            FeatureContribution("Flow Duration", 1200.0, 0.14, "increases_risk"),
        ],
    )
    _sample_text = _template_fallback(_sample_exp, sample_profile)
    st.markdown(
        "<div style='background:#ff4b4b20;border-left:5px solid #ff4b4b;"
        "padding:14px;border-radius:6px;margin-bottom:12px'>"
        "<h4 style='color:#ff4b4b;margin:0'>HIGH — ATTACK DETECTED</h4>"
        "<p style='margin:4px 0 0'>Confidence: 91.0% | Profile: General Employee</p></div>",
        unsafe_allow_html=True,
    )
    st.write(_sample_text)
    st.info(
        "The system is highly confident in this detection. "
        "The network pattern strongly matches known attack signatures."
    )
    st.markdown("**Recommended actions:**")
    st.write("1. Stop using your work device and disconnect from the network.")
    st.write("2. Contact your IT helpdesk or support team immediately.")
    st.write("3. Do not open emails or attachments until cleared.")

    st.divider()
    st.subheader("Survey Questions")

    with st.form("user_study_form", clear_on_submit=True):
        q1 = st.radio(
            "Q1. Did you understand what happened?",
            ["Yes", "Somewhat", "No"],
            horizontal=True,
        )
        q2 = st.radio(
            "Q2. Did you understand what to do next?",
            ["Yes", "Somewhat", "No"],
            horizontal=True,
        )
        q3 = st.slider(
            "Q3. How clear was the language? (1 = very confusing, 5 = very clear)",
            min_value=1, max_value=5, value=3,
        )
        q4 = st.radio(
            "Q4. Would this alert make you take action?",
            ["Yes", "Maybe", "No"],
            horizontal=True,
        )
        q5 = st.radio(
            "Q5. What is your technical background?",
            ["Non-technical", "Some tech knowledge", "IT professional"],
            horizontal=True,
        )
        submitted = st.form_submit_button("Submit Response", type="primary", use_container_width=True)

    if submitted:
        try:
            _RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
            write_header = not _RESPONSES_PATH.exists()
            with open(_RESPONSES_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["timestamp", "q1_understood_what", "q2_understood_action",
                                     "q3_clarity", "q4_would_act", "q5_background"])
                writer.writerow([
                    datetime.datetime.now().isoformat(timespec="seconds"),
                    q1, q2, q3, q4, q5,
                ])
            st.success("Thank you! Your response has been recorded.")
        except Exception as _e:
            st.warning(f"Could not save response: {_e}")

    # Running count (no individual data shown)
    if _RESPONSES_PATH.exists():
        try:
            with open(_RESPONSES_PATH, encoding="utf-8") as f:
                count = sum(1 for _ in csv.reader(f)) - 1
            st.metric("Responses collected so far", max(0, count))
        except Exception:
            st.metric("Responses collected so far", 0)
    else:
        st.metric("Responses collected so far", 0)

# ===========================================================================
# TAB 5 — About
# ===========================================================================
with tab_about:
    st.header("About This System")
    st.markdown("""
## XAI Cybersecurity Alert System — v2.0

**Status:** Patent-Pending Research

🌐 **Live Demo:** [xai-cybersecurity-alert-system-flmwn94cvuxsyllsvudhsl.streamlit.app](https://xai-cybersecurity-alert-system-flmwn94cvuxsyllsvudhsl.streamlit.app/)

💻 **Source Code:** [github.com/Irin-Akther/xai-cybersecurity-alert-system](https://github.com/Irin-Akther/xai-cybersecurity-alert-system)

### 10 User Personas

| Persona | Level | Explanation Style |
|---------|-------|-------------------|
| 👦 Kid | HOME | Super simple, fun analogies |
| 🧑 Teenager | HOME | Casual, relatable language |
| 🏠 Housewife | HOME | Home and family analogies |
| 🛒 Cashier | HOME | Simple, one-action guidance |
| 💼 General Employee | HOME | Plain office language |
| 🏢 Business Owner | SMB | Business impact focus |
| 🎓 Student | SMB | Educational with learning tips |
| 👔 Executive / Manager | SMB | Executive summary style |
| 📋 Compliance / Auditor | ADMIN | Regulatory and audit language |
| 🔐 Security Analyst | ADMIN | Full forensic technical detail |

### 6 Pre-Built Scenarios

| Scenario | Type |
|----------|------|
| DDoS Attack | Volumetric flood |
| Port Scan | Network discovery |
| Normal HTTPS Traffic | Benign baseline |
| SSH Brute Force | Credential attack |
| Malware C2 Beacon | Command & Control |
| Suspicious Link / Redirect | Initial access |

### Architecture

| Component | Technology |
|-----------|-----------|
| Threat Detector | Random Forest (scikit-learn) on CICIDS2017 |
| XAI Explainer | SHAP TreeExplainer |
| User Profiler | 10-persona literacy classifier |
| NLG Module | Ollama (Mistral/LLaMA 3) — fully on-device |
| Dashboard | Streamlit |

### Privacy Design
All inference runs locally. No data is sent to external APIs.

---
*Developed by Irin — Cybersecurity Research*
    """)
