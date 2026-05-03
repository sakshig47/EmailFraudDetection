# =============================================================================
# app/app.py  —  Streamlit Web Application
# Email Fraud Detection — Spam & Phishing Classifier
# Run: streamlit run app/app.py
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data.preprocess import preprocess_text
from src.config import MODEL_PATH, LABEL_NAMES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Fraud Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .result-safe {
        background: linear-gradient(135deg, #0f5132, #198754);
        color: white; border-radius: 12px; padding: 20px 28px;
        font-size: 1.4rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 20px rgba(25,135,84,0.4);
    }
    .result-spam {
        background: linear-gradient(135deg, #7f1d1d, #dc2626);
        color: white; border-radius: 12px; padding: 20px 28px;
        font-size: 1.4rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 20px rgba(220,38,38,0.4);
    }
    .metric-card {
        background: #1e1e2e; border: 1px solid #333; border-radius: 10px;
        padding: 14px 20px; text-align: center; color: #e2e8f0;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; }
    .metric-card .label { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
    .stTextArea textarea { font-family: monospace; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained sklearn Pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model not found at `{MODEL_PATH}`.\n\n"
            "Run `python main.py` first to train and save the model."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


def classify_email(raw_text: str, pipeline) -> dict:
    """Preprocess text and return prediction + confidence."""
    clean  = preprocess_text(raw_text)
    proba  = pipeline.predict_proba([clean])[0]
    idx    = int(np.argmax(proba))
    return {
        "label":      LABEL_NAMES[idx],
        "label_idx":  idx,
        "confidence": float(proba[idx]),
        "spam_prob":  float(proba[1]),
        "ham_prob":   float(proba[0]),
    }


def gauge_chart(spam_prob: float) -> plt.Figure:
    """Draw a minimal probability gauge."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")

    # Background bar
    ax.barh(0, 1, height=0.4, color="#1e1e2e", edgecolor="none")
    # Filled bar
    color = "#dc2626" if spam_prob > 0.5 else "#22c55e"
    ax.barh(0, spam_prob, height=0.4, color=color, edgecolor="none")

    ax.axvline(0.5, color="#94a3b8", lw=1, ls="--", alpha=0.6)
    ax.text(spam_prob, 0, f" {spam_prob*100:.1f}%", va="center",
            color="white", fontsize=13, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="#94a3b8", fontsize=9)
    ax.spines[:].set_visible(False)
    ax.tick_params(colors="#94a3b8")
    ax.set_title("Spam/Phishing Probability", color="#94a3b8", fontsize=10, pad=8)

    plt.tight_layout()
    return fig


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## 🛡️ About this Tool")
    st.markdown("""
This tool analyzes the content of an email and determines 
whether it is safe or potentially fraudulent.
- 📩 **It scans the message for suspicious language, patterns, and intent**
- ⚠️ **It detects possible spam, phishing attempts, or scam messages**
- ✅ **It tells you clearly whether the email is Legitimate or Suspicious**
- 🔍 **It helps users avoid clicking harmful links or sharing sensitive information**

- ✅ **Legitimate** — safe to read
- 🚨 **Spam / Phishing** — potential fraud

    """)
    st.divider()
    st.markdown("### 🧪 Try example emails")

    examples = {
        "💸 Prize scam": (
            "WINNER!! You have been selected to receive a £900 prize! "
            "Call 09061701461 to claim. Visit http://freeprizeclaim.win"
        ),
        "🎣 Phishing": (
            "Dear Customer, your PayPal account has been limited. "
            "Please verify at http://paypa1-secure.net/verify immediately."
        ),
        "📧 Legitimate": (
            "Hi Sarah, just confirming our 10am meeting on Thursday. "
            "I've attached the agenda — let me know if you'd like to add anything."
        ),
        "📢 Marketing spam": (
            "Act now! Limited time offer — buy one get one FREE on all items. "
            "Click here to shop: www.dealsnow.biz. Unsubscribe below."
        ),
    }
    selected = st.selectbox("Select an example:", list(examples.keys()))
    load_example = st.button("📋 Load Example")


# =============================================================================
# Main content
# =============================================================================
st.markdown("# 🛡️ Email Fraud Detector")
st.markdown("""This tool analyzes the content of an email and determines
            whether it is safe or potentially fraudulent.""")
st.divider()

pipeline = load_model()

# Populate textarea with example if requested
initial_text = examples[selected] if load_example else ""

email_input = st.text_area(
    "📨 Paste email content here:",
    value=initial_text,
    height=200,
    placeholder="Paste the body of any email to check if it's spam or phishing …",
)

col1, col2 = st.columns([1, 4])
with col1:
    analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑 Clear", use_container_width=True)

if clear_btn:
    st.rerun()

st.divider()

# =============================================================================
# Results
# =============================================================================
if analyse_btn and email_input.strip():
    with st.spinner("Analysing …"):
        result = classify_email(email_input, pipeline)

    is_spam = result["label_idx"] == 1
    emoji   = "🚨" if is_spam else "✅"
    css_cls = "result-spam" if is_spam else "result-safe"

    st.markdown(
        f'<div class="{css_cls}">{emoji}  {result["label"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Metrics row
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Spam Probability",  f"{result['spam_prob']*100:.1f}%")
    with mc2:
        st.metric("Ham Probability",   f"{result['ham_prob']*100:.1f}%")
    with mc3:
        st.metric("Model Confidence",  f"{result['confidence']*100:.1f}%")

    # Gauge chart
    fig = gauge_chart(result["spam_prob"])
    st.pyplot(fig, use_container_width=True)

    # Preprocessed text (expandable)
    with st.expander("🔬 Show preprocessed text"):
        clean = preprocess_text(email_input)
        st.code(clean, language="text")

    # Warning for borderline cases
    if 0.4 < result["spam_prob"] < 0.6:
        st.warning(
            "⚠️ **Borderline result** — confidence is low. "
            "Review the email manually before taking action."
        )

elif analyse_btn and not email_input.strip():
    st.warning("Please paste some email text before analysing.")

else:
    # Placeholder state
    st.info(
        "👈 Paste email text in the box above and click **Analyse**, "
        "or load one of the example emails from the sidebar."
    )

# =============================================================================
# Footer
# =============================================================================
st.divider()
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:0.8rem;'>"
    "Email Fraud Detection · Internship Portfolio Project · "
    "Built with Streamlit + scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
