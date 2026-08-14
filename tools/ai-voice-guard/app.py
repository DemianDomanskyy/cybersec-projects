"""AI Voice Guard — detects whether an audio clip is human or AI-generated speech."""

import os
import tempfile
import time
from datetime import datetime

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

st.set_page_config(
    page_title="AI Voice Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
    --bg: #0b0f14;
    --panel: #111820;
    --panel-border: #1e2a35;
    --accent: #39f6c0;
    --accent-dim: #1c8f74;
    --danger: #ff4d6d;
    --danger-dim: #7a1e2e;
    --text: #e6f1ef;
    --text-dim: #7d94a3;
}

html, body, [class*="css"] { font-family: 'Consolas', 'SF Mono', monospace; }
.stApp { background: radial-gradient(circle at 20% 0%, #0f1720 0%, var(--bg) 55%); }

[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
}

h1, h2, h3 { color: var(--text) !important; letter-spacing: 0.02em; }

.brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.2rem;
}
.brand-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
}
.brand-sub {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}

.stat-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
}
.stat-label { color: var(--text-dim); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
.stat-value { color: var(--accent); font-size: 1.15rem; font-weight: 700; }

.verdict-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    border: 1px solid;
    animation: fadein 0.4s ease;
}
.verdict-real {
    background: linear-gradient(135deg, rgba(57,246,192,0.10), rgba(57,246,192,0.02));
    border-color: var(--accent-dim);
}
.verdict-fake {
    background: linear-gradient(135deg, rgba(255,77,109,0.12), rgba(255,77,109,0.02));
    border-color: var(--danger-dim);
}
.verdict-label { font-size: 1.8rem; font-weight: 800; letter-spacing: 0.03em; }
.verdict-real .verdict-label { color: var(--accent); }
.verdict-fake .verdict-label { color: var(--danger); }
.verdict-sub { color: var(--text-dim); font-size: 0.95rem; margin-top: 0.3rem; }

@keyframes fadein { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

.history-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0.7rem;
    border-bottom: 1px solid var(--panel-border);
    font-size: 0.85rem;
    color: var(--text-dim);
}
.history-row span.tag-real { color: var(--accent); font-weight: 600; }
.history-row span.tag-fake { color: var(--danger); font-weight: 600; }

section[data-testid="stFileUploaderDropzone"], div[data-testid="stAudioInput"] {
    background: var(--panel) !important;
    border: 1px dashed var(--panel-border) !important;
    border-radius: 12px !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: var(--panel);
    border-radius: 8px 8px 0 0;
    color: var(--text-dim);
}
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return feature_extractor, model


def analyze(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
    finally:
        os.unlink(tmp_path)

    feature_extractor, model = load_model()
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    fake_prob = float(probs[1])
    return {
        "is_fake": fake_prob > 0.5,
        "confidence": fake_prob if fake_prob > 0.5 else 1 - fake_prob,
        "fake_prob": fake_prob,
        "audio": audio,
        "sr": sr,
        "duration": len(audio) / sr,
    }


def render_waveform(audio: np.ndarray, sr: int, is_fake: bool):
    color = "#ff4d6d" if is_fake else "#39f6c0"
    step = max(1, len(audio) // 2000)
    y = audio[::step]
    x = np.arange(len(y)) * step / sr

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, line=dict(color=color, width=1),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.15),
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#7d94a3", title="seconds"),
        yaxis=dict(showgrid=False, color="#7d94a3", showticklabels=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def render_gauge(fake_prob: float):
    color = "#ff4d6d" if fake_prob > 0.5 else "#39f6c0"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fake_prob * 100,
        number={"suffix": "%", "font": {"color": color, "size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#7d94a3", "tickfont": {"color": "#7d94a3"}},
            "bar": {"color": color},
            "bgcolor": "#111820",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(57,246,192,0.10)"},
                {"range": [50, 100], "color": "rgba(255,77,109,0.10)"},
            ],
            "threshold": {"line": {"color": "#e6f1ef", "width": 2}, "thickness": 0.8, "value": 50},
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d94a3"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_verdict(result: dict, source_label: str):
    css_class = "verdict-fake" if result["is_fake"] else "verdict-real"
    label = "SOUNDS AI-GENERATED" if result["is_fake"] else "SOUNDS HUMAN"
    icon = "⚠️" if result["is_fake"] else "✅"
    st.markdown(
        f"""<div class="verdict-card {css_class}">
                <div class="verdict-label">{icon} {label}</div>
                <div class="verdict-sub">{result['confidence']*100:.1f}% confidence · {result['duration']:.1f}s clip · {source_label}</div>
            </div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption("FAKE-VOICE PROBABILITY")
        render_gauge(result["fake_prob"])
    with col2:
        st.caption("WAVEFORM")
        render_waveform(result["audio"], result["sr"], result["is_fake"])


def log_history(source_label: str, result: dict):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source_label,
        "is_fake": result["is_fake"],
        "confidence": result["confidence"],
    })
    st.session_state.history = st.session_state.history[:20]


with st.sidebar:
    st.markdown(
        """<div class="brand">
                <div style="font-size:1.8rem;">🛡️</div>
                <div class="brand-title">AI Voice Guard</div>
           </div>
           <div class="brand-sub">Checks whether a clip is a real voice or AI-generated</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="stat-card"><div class="stat-label">Model</div>
               <div class="stat-value" style="font-size:0.85rem;">wav2vec2-XLSR</div></div>
           <div class="stat-card"><div class="stat-label">Validation accuracy</div>
               <div class="stat-value">97.9%</div></div>
           <div class="stat-card"><div class="stat-label">ROC-AUC</div>
               <div class="stat-value">0.998</div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("SCAN HISTORY")
    history = st.session_state.get("history", [])
    if not history:
        st.caption("No scans yet this session.")
    else:
        for h in history:
            tag_class = "tag-fake" if h["is_fake"] else "tag-real"
            tag_text = "AI" if h["is_fake"] else "HUMAN"
            st.markdown(
                f"""<div class="history-row">
                        <span>{h['time']} · {h['source']}</span>
                        <span class="{tag_class}">{tag_text} {h['confidence']*100:.0f}%</span>
                    </div>""",
                unsafe_allow_html=True,
            )

st.title("Voice Scanner")
st.caption("Upload a clip or record one — it'll tell you whether the voice sounds real or synthetic.")

with st.spinner("Loading model…"):
    load_model()

tab_upload, tab_live = st.tabs(["Upload", "Record"])

with tab_upload:
    uploaded = st.file_uploader("WAV, MP3, FLAC, M4A, or OGG", type=["wav", "mp3", "flac", "m4a", "ogg"])
    if uploaded is not None:
        st.audio(uploaded)
        with st.spinner("Analyzing…"):
            start = time.time()
            result = analyze(uploaded.getvalue())
            elapsed = time.time() - start
        render_verdict(result, uploaded.name)
        st.caption(f"{elapsed:.2f}s")
        log_history(uploaded.name, result)

with tab_live:
    st.write("Record a few seconds from your mic.")
    recorded = st.audio_input("Hit record, say something, then stop")
    if recorded is not None:
        with st.spinner("Analyzing waveform…"):
            start = time.time()
            result = analyze(recorded.getvalue())
            elapsed = time.time() - start
        render_verdict(result, "mic recording")
        st.caption(f"{elapsed:.2f}s")
        log_history("mic recording", result)

st.markdown("---")
st.caption(
    "Model: wav2vec2, fine-tuned on ElevenLabs, Amazon Polly, and Speechify samples. "
    "It's a decent signal, not a verdict — don't bet anything important on it alone."
)
