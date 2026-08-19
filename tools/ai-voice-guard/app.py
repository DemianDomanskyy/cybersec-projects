"""AI Voice Guard — detects whether an audio clip is human or AI-generated speech."""

import hashlib
import json
import os
import secrets
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

st.set_page_config(
    page_title="AI Voice Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0a0e13;
    --panel: #11171f;
    --panel-2: #161d27;
    --panel-border: #212b38;
    --accent: #39f6c0;
    --accent-dim: #1c8f74;
    --accent-glow: rgba(57, 246, 192, 0.35);
    --danger: #ff5577;
    --danger-dim: #7a1e2e;
    --danger-glow: rgba(255, 85, 119, 0.3);
    --text: #eaf3f1;
    --text-dim: #83a0af;
    --radius: 14px;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'Consolas', 'SF Mono', monospace;
}
.stApp {
    background:
        radial-gradient(circle at 15% -10%, rgba(57,246,192,0.06) 0%, transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(57,246,192,0.04) 0%, transparent 35%),
        var(--bg);
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--panel-border); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--panel) 0%, var(--bg) 100%);
    border-right: 1px solid var(--panel-border);
}

h1, h2, h3 { color: var(--text) !important; letter-spacing: 0.02em; font-weight: 800 !important; }
p, span, label, .stMarkdown { color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--text-dim) !important; }

.brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.2rem;
}
.brand-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: 0.01em;
}
.brand-sub {
    color: var(--text-dim);
    font-size: 0.82rem;
    margin-bottom: 1.4rem;
    line-height: 1.4;
}

/* --- Login page --- */
.login-brand {
    text-align: center;
    margin: 3.5rem 0 0.5rem;
}
.login-shield {
    font-size: 2.6rem;
    filter: drop-shadow(0 0 18px var(--accent-glow));
    margin-bottom: 0.3rem;
}
.login-brand .brand-title { font-size: 1.9rem; display: block; }
.login-sub { color: var(--text-dim); margin-top: 0.3rem; font-size: 0.95rem; }
/* The login form is the only bordered st.container() in the app, so this
   selector is safe to target globally. */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border-radius: var(--radius) !important;
    box-shadow: 0 20px 50px -20px rgba(0,0,0,0.6);
    margin-top: 1rem;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: var(--panel-border) !important;
    border-radius: var(--radius) !important;
}

/* --- Sidebar stat cards --- */
.stat-card {
    background: var(--panel-2);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s ease;
}
.stat-card:hover { border-color: var(--accent-dim); }
.stat-label { color: var(--text-dim); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.09em; }
.stat-value { color: var(--accent); font-size: 1.05rem; font-weight: 700; }

/* --- Verdict card --- */
.verdict-card {
    border-radius: var(--radius);
    padding: 2.2rem 2rem;
    text-align: center;
    margin: 1rem 0;
    border: 1px solid;
    animation: fadein 0.45s ease;
    position: relative;
    overflow: hidden;
}
.verdict-real {
    background: linear-gradient(135deg, rgba(57,246,192,0.12), rgba(57,246,192,0.02));
    border-color: var(--accent-dim);
    box-shadow: 0 0 60px -25px var(--accent-glow);
}
.verdict-fake {
    background: linear-gradient(135deg, rgba(255,85,119,0.14), rgba(255,85,119,0.02));
    border-color: var(--danger-dim);
    box-shadow: 0 0 60px -25px var(--danger-glow);
}
.verdict-label { font-size: 1.9rem; font-weight: 800; letter-spacing: 0.02em; }
.verdict-real .verdict-label { color: var(--accent); }
.verdict-fake .verdict-label { color: var(--danger); }
.verdict-sub { color: var(--text-dim); font-size: 0.95rem; margin-top: 0.4rem; }

/* --- Multi-model consensus + comparison --- */
.consensus-banner {
    border-radius: 12px;
    padding: 1.05rem 1.4rem;
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
    padding: 1rem 1.1rem;
    height: 100%;
    transition: transform 0.15s ease, border-color 0.15s ease;
    border-top: 3px solid var(--panel-border);
}
.model-card:hover { transform: translateY(-2px); }
.model-card-name {
    font-size: 0.74rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    min-height: 2.2em;
}
.model-card-verdict { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.3rem; }
.model-card-verdict.fake { color: var(--danger); }
.model-card-verdict.real { color: var(--accent); }
.model-card-bar-bg { background: #1a232e; border-radius: 6px; height: 7px; overflow: hidden; margin-top: 0.6rem; }
.model-card-bar-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease; }

@keyframes fadein { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

/* --- Scan history --- */
.history-row {
    display: flex;
    justify-content: space-between;
    padding: 0.55rem 0.7rem;
    border-bottom: 1px solid var(--panel-border);
    font-size: 0.83rem;
    color: var(--text-dim);
}
.history-row:last-child { border-bottom: none; }
.history-row span.tag-real { color: var(--accent); font-weight: 700; }
.history-row span.tag-fake { color: var(--danger); font-weight: 700; }

/* --- Upload / record widgets --- */
section[data-testid="stFileUploaderDropzone"], div[data-testid="stAudioInput"] {
    background: var(--panel) !important;
    border: 1px dashed var(--panel-border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.15s ease;
}
section[data-testid="stFileUploaderDropzone"]:hover, div[data-testid="stAudioInput"]:hover {
    border-color: var(--accent-dim) !important;
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: var(--panel);
    border-radius: 8px 8px 0 0;
    color: var(--text-dim);
    font-weight: 600;
}
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

/* --- Buttons --- */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
    border-radius: 10px !important;
    border: 1px solid var(--panel-border) !important;
    background: var(--panel-2) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    transition: border-color 0.15s ease, transform 0.1s ease !important;
}
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-1px);
}
.stFormSubmitButton button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, var(--accent-dim), var(--accent)) !important;
    color: #06231b !important;
    border: none !important;
}
.stFormSubmitButton button[kind="primaryFormSubmit"]:hover { color: #06231b !important; }

/* --- Multiselect tags: swap Streamlit's alarm-red default for the accent color --- */
span[data-tag] {
    background: rgba(57,246,192,0.16) !important;
    border-radius: 6px !important;
}
span[data-tag] span, span[data-tag] button { color: var(--accent) !important; }
span[data-tag] svg { stroke: var(--accent) !important; }

[data-testid="stTextInput"] input {
    background: var(--panel-2) !important;
    border-color: var(--panel-border) !important;
    color: var(--text) !important;
}
[data-testid="stTextInput"] input:focus { border-color: var(--accent) !important; }

hr { border-color: var(--panel-border) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Auth — a lightweight gate, not real access control. Good enough to keep
# casual visitors out of a demo; not a substitute for real auth if this ever
# handles anything sensitive.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _generated_demo_password() -> str:
    """A random password generated once per running server process.

    Never hardcoded, never committed — printed only to the server's own
    console log, so whoever has access to the machine/logs running the app
    can read it, but no credential ever sits in source control.
    """
    pw = secrets.token_urlsafe(9)
    print(
        f"\n[AI Voice Guard] No APP_PASSWORD configured — generated a "
        f"one-time local login password: {pw}\n",
        flush=True,
    )
    return pw


def _resolve_credentials():
    username = os.environ.get("APP_USERNAME", "admin")
    if "APP_PASSWORD_HASH" in os.environ:
        return username, os.environ["APP_PASSWORD_HASH"], False
    if "APP_PASSWORD" in os.environ:
        return username, hashlib.sha256(os.environ["APP_PASSWORD"].encode()).hexdigest(), False
    generated = _generated_demo_password()
    return username, hashlib.sha256(generated.encode()).hexdigest(), True


def require_login() -> bool:
    if st.session_state.get("authenticated"):
        return True

    username, password_hash, using_generated = _resolve_credentials()

    st.markdown(
        """<div class="login-brand">
                <div class="login-shield">🛡️</div>
                <div class="brand-title">AI Voice Guard</div>
                <p class="login-sub">Sign in to continue</p>
           </div>""",
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        with st.container(border=True):
            if using_generated:
                st.warning(
                    f"No `APP_PASSWORD` configured — using username `{username}` with a "
                    "one-time password printed to the server's console log for this run. "
                    "Set `APP_USERNAME`/`APP_PASSWORD` env vars for a stable login "
                    "before deploying this anywhere public.",
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
