"""AI Voice Guard — detects whether an audio clip is human or AI-generated speech."""

import hashlib
import io
import json
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

MODELS = {
    "wav2vec2-XLSR (deepfake voice)": "garystafford/wav2vec2-deepfake-voice-detector",
    "wav2vec2 (deepfake audio)": "mo-thecreator/Deepfake-audio-detection",
    "wav2vec2-XLSR-large (deepfake audio)": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
    "wav2vec2 (deepfake audio v2)": "MelodyMachine/Deepfake-audio-detection-V2",
}
DEFAULT_MODEL = "wav2vec2-XLSR (deepfake voice)"

DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "REDACTED"

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
.stat-value { color: var(--accent); font-size: 1.05rem; font-weight: 700; }

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

.consensus-banner {
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0 1.2rem;
    border: 1px solid var(--panel-border);
    background: var(--panel);
    font-size: 1rem;
    color: var(--text);
}

.model-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    height: 100%;
}
.model-card-name { font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
.model-card-verdict { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.3rem; }
.model-card-verdict.fake { color: var(--danger); }
.model-card-verdict.real { color: var(--accent); }
.model-card-bar-bg { background: #1e2a35; border-radius: 6px; height: 8px; overflow: hidden; margin-top: 0.5rem; }
.model-card-bar-fill { height: 100%; border-radius: 6px; }

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


# ---------------------------------------------------------------------------
# Auth — a lightweight gate, not real access control. Good enough to keep
# casual visitors out of a demo; not a substitute for real auth if this ever
# handles anything sensitive.
# ---------------------------------------------------------------------------

def _resolve_credentials():
    username = os.environ.get("APP_USERNAME", DEFAULT_USERNAME)
    using_default = True
    if "APP_PASSWORD_HASH" in os.environ:
        password_hash = os.environ["APP_PASSWORD_HASH"]
        using_default = False
    elif "APP_PASSWORD" in os.environ:
        password_hash = hashlib.sha256(os.environ["APP_PASSWORD"].encode()).hexdigest()
        using_default = False
    else:
        password_hash = hashlib.sha256(DEFAULT_PASSWORD.encode()).hexdigest()
    return username, password_hash, using_default


def require_login() -> bool:
    if st.session_state.get("authenticated"):
        return True

    username, password_hash, using_default = _resolve_credentials()

    st.markdown(
        """<div class="brand" style="margin-top:3rem; justify-content:center;">
                <div style="font-size:2rem;">🛡️</div>
                <div class="brand-title" style="font-size:2rem;">AI Voice Guard</div>
           </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:var(--text-dim);'>Sign in to continue</p>",
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        if using_default:
            st.warning(
                f"Using default demo credentials (`{DEFAULT_USERNAME}` / `{DEFAULT_PASSWORD}`). "
                "Set `APP_USERNAME` and `APP_PASSWORD` environment variables before deploying "
                "this anywhere public.",
                icon="⚠️",
            )
        with st.form("login_form"):
            user_in = st.text_input("Username")
            pass_in = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            pass_hash_in = hashlib.sha256(pass_in.encode()).hexdigest()
            if user_in == username and pass_hash_in == password_hash:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Wrong username or password.")

    return False


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_id: str):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()
    return feature_extractor, model


def _fake_label_index(id2label: dict) -> int:
    for idx, label in id2label.items():
        if any(word in label.lower() for word in ("fake", "spoof", "synthetic", "ai")):
            return int(idx)
    # Fallback: assume index 1 if no label name matches (shouldn't happen for
    # the models registered above, but avoids a hard crash on a new one).
    return 1


def decode_audio(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
    finally:
        os.unlink(tmp_path)
    return audio, sr


def run_model(audio: np.ndarray, model_name: str) -> dict:
    model_id = MODELS[model_name]
    feature_extractor, model = load_model(model_id)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    fake_idx = _fake_label_index(model.config.id2label)
    fake_prob = float(probs[fake_idx])
    return {
        "model_name": model_name,
        "model_id": model_id,
        "is_fake": fake_prob > 0.5,
        "confidence": fake_prob if fake_prob > 0.5 else 1 - fake_prob,
        "fake_prob": fake_prob,
    }


def analyze(audio_bytes: bytes, model_names: list[str]) -> dict:
    audio, sr = decode_audio(audio_bytes)
    results = [run_model(audio, name) for name in model_names]
    return {
        "results": results,
        "audio": audio,
        "sr": sr,
        "duration": len(audio) / sr,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


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


def render_single_verdict(result: dict, source_label: str, duration: float, audio, sr):
    css_class = "verdict-fake" if result["is_fake"] else "verdict-real"
    label = "SOUNDS AI-GENERATED" if result["is_fake"] else "SOUNDS HUMAN"
    icon = "⚠️" if result["is_fake"] else "✅"
    st.markdown(
        f"""<div class="verdict-card {css_class}">
                <div class="verdict-label">{icon} {label}</div>
                <div class="verdict-sub">{result['confidence']*100:.1f}% confidence · {duration:.1f}s clip · {source_label}</div>
            </div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption("FAKE-VOICE PROBABILITY")
        render_gauge(result["fake_prob"])
    with col2:
        st.caption("WAVEFORM")
        render_waveform(audio, sr, result["is_fake"])


def render_model_grid(results: list[dict]):
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        with col:
            tag = "fake" if r["is_fake"] else "real"
            color = "#ff4d6d" if r["is_fake"] else "#39f6c0"
            verdict_text = "AI-generated" if r["is_fake"] else "Human"
            st.markdown(
                f"""<div class="model-card">
                        <div class="model-card-name">{r['model_name']}</div>
                        <div class="model-card-verdict {tag}">{verdict_text}</div>
                        <div style="color:var(--text-dim); font-size:0.85rem;">{r['confidence']*100:.1f}% confidence</div>
                        <div class="model-card-bar-bg">
                            <div class="model-card-bar-fill" style="width:{r['fake_prob']*100:.1f}%; background:{color};"></div>
                        </div>
                    </div>""",
                unsafe_allow_html=True,
            )


def render_consensus(results: list[dict]):
    fake_votes = sum(1 for r in results if r["is_fake"])
    total = len(results)
    avg_fake_prob = sum(r["fake_prob"] for r in results) / total
    majority_fake = fake_votes > total / 2
    label = "AI-generated" if majority_fake else "Human"
    icon = "⚠️" if majority_fake else "✅"
    st.markdown(
        f"""<div class="consensus-banner">
                {icon} <b>{fake_votes} of {total} models</b> say <b>{label}</b>
                · average fake-voice probability {avg_fake_prob*100:.1f}%
            </div>""",
        unsafe_allow_html=True,
    )


def render_result(analysis: dict, source_label: str):
    results = analysis["results"]
    if len(results) == 1:
        render_single_verdict(results[0], source_label, analysis["duration"], analysis["audio"], analysis["sr"])
    else:
        render_consensus(results)
        render_model_grid(results)
        st.caption("WAVEFORM")
        render_waveform(analysis["audio"], analysis["sr"], results[0]["is_fake"])

    report = {
        "source": source_label,
        "duration_seconds": round(analysis["duration"], 2),
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "results": [
            {
                "model": r["model_name"],
                "model_id": r["model_id"],
                "verdict": "ai-generated" if r["is_fake"] else "human",
                "confidence": round(r["confidence"], 4),
                "fake_probability": round(r["fake_prob"], 4),
            }
            for r in results
        ],
    }
    st.download_button(
        "Download report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"voice-guard-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
        mime="application/json",
    )


def log_history(source_label: str, analysis: dict):
    if "history" not in st.session_state:
        st.session_state.history = []
    results = analysis["results"]
    fake_votes = sum(1 for r in results if r["is_fake"])
    is_fake = fake_votes > len(results) / 2
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source_label,
        "is_fake": is_fake,
        "confidence": avg_confidence,
        "models": len(results),
    })
    st.session_state.history = st.session_state.history[:20]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

if not require_login():
    st.stop()

with st.sidebar:
    st.markdown(
        """<div class="brand">
                <div style="font-size:1.8rem;">🛡️</div>
                <div class="brand-title">AI Voice Guard</div>
           </div>
           <div class="brand-sub">Checks whether a clip is a real voice or AI-generated</div>""",
        unsafe_allow_html=True,
    )

    selected_models = st.multiselect(
        "Models to run",
        options=list(MODELS.keys()),
        default=[DEFAULT_MODEL],
        help="Pick one for a fast single verdict, or several for a consensus comparison.",
    )

    st.markdown(
        f"""<div class="stat-card"><div class="stat-label">Active models</div>
               <div class="stat-value">{len(selected_models)}</div></div>
           <div class="stat-card"><div class="stat-label">Architecture</div>
               <div class="stat-value">wav2vec2 family</div></div>""",
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
            models_note = f" · {h['models']} models" if h.get("models", 1) > 1 else ""
            st.markdown(
                f"""<div class="history-row">
                        <span>{h['time']} · {h['source']}{models_note}</span>
                        <span class="{tag_class}">{tag_text} {h['confidence']*100:.0f}%</span>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    if st.button("Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.title("Voice Scanner")
st.caption("Upload a clip or record one — it'll tell you whether the voice sounds real or synthetic.")

if not selected_models:
    st.info("Pick at least one model in the sidebar to run an analysis.")
    st.stop()

with st.spinner(f"Loading {len(selected_models)} model(s)…"):
    for name in selected_models:
        load_model(MODELS[name])

tab_upload, tab_live = st.tabs(["Upload", "Record"])

with tab_upload:
    uploaded = st.file_uploader("WAV, MP3, FLAC, M4A, or OGG", type=["wav", "mp3", "flac", "m4a", "ogg"])
    if uploaded is not None:
        st.audio(uploaded)
        with st.spinner("Analyzing…"):
            start = time.time()
            analysis = analyze(uploaded.getvalue(), selected_models)
            elapsed = time.time() - start
        render_result(analysis, uploaded.name)
        st.caption(f"{elapsed:.2f}s")
        log_history(uploaded.name, analysis)

with tab_live:
    st.write("Record a few seconds from your mic.")
    recorded = st.audio_input("Hit record, say something, then stop")
    if recorded is not None:
        with st.spinner("Analyzing…"):
            start = time.time()
            analysis = analyze(recorded.getvalue(), selected_models)
            elapsed = time.time() - start
        render_result(analysis, "mic recording")
        st.caption(f"{elapsed:.2f}s")
        log_history("mic recording", analysis)

st.markdown("---")
st.caption(
    "Multiple independent wav2vec2-family classifiers, each fine-tuned separately on "
    "AI voice/deepfake audio datasets. Treat any result as a signal, not a verdict — "
    "don't bet anything important on it alone."
)
