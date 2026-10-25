"""ML inference: model loading, per-clip and per-window scoring."""

import os
import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Sentinel model_id (not a HuggingFace repo id) that routes run_model()/
# load_model() to the local classical-ML classifier below instead of a
# wav2vec2 transformer.
CUSTOM_MODEL_ID = "custom:voice-conversion-detector"
CUSTOM_MODEL_PATH = Path(__file__).parent / "custom_model" / "voice_conversion_classifier.joblib"
CUSTOM_WINDOW_S = 1.0  # must match test-samples/train_classifier.py's WINDOW_S

MODELS = {
    "wav2vec2-XLSR (deepfake voice)": "garystafford/wav2vec2-deepfake-voice-detector",
    "wav2vec2 (deepfake audio)": "mo-thecreator/Deepfake-audio-detection",
    "wav2vec2-XLSR-large (deepfake audio)": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
    "wav2vec2 (deepfake audio v2)": "MelodyMachine/Deepfake-audio-detection-V2",
    "Voice-conversion detector (custom, local)": CUSTOM_MODEL_ID,
}
DEFAULT_MODEL = "wav2vec2-XLSR (deepfake voice)"

_model_cache: dict[str, tuple] = {}


def load_model(model_id: str):
    if model_id not in _model_cache:
        if model_id == CUSTOM_MODEL_ID:
            _model_cache[model_id] = joblib.load(CUSTOM_MODEL_PATH)
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            model = AutoModelForAudioClassification.from_pretrained(model_id)
            model.eval()
            _model_cache[model_id] = (feature_extractor, model)
    return _model_cache[model_id]


def _extract_custom_features(audio: np.ndarray, sr: int, window_s: float = CUSTOM_WINDOW_S) -> list[list[float]]:
    """One feature row per window_s window: chroma/RMS/spectral/MFCC means,
    the same 26 columns (in the same order) used in
    test-samples/train_classifier.py -- must stay in sync with that script.
    """
    window = int(window_s * sr)
    if len(audio) < window:
        audio = np.pad(audio, (0, window - len(audio)))
    hop = window  # non-overlapping, matches training
    rows = []
    start = 0
    while start + window <= len(audio):
        seg = audio[start:start + window]
        chroma = np.mean(librosa.feature.chroma_stft(y=seg, sr=sr))
        rms = np.mean(librosa.feature.rms(y=seg))
        centroid = np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=seg))
        mfcc = np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20), axis=1)
        rows.append([chroma, rms, centroid, bandwidth, rolloff, zcr, *mfcc])
        start += hop
    return rows


def _run_custom_model(audio: np.ndarray, sr: int = 16000) -> dict:
    bundle = load_model(CUSTOM_MODEL_ID)
    rows = _extract_custom_features(audio, sr)
    X = bundle["scaler"].transform(rows)
    # predict_proba[:, 1] = P(fake) per window; average across windows for a
    # single clip-level score, same spirit as the wav2vec2 path's single logits.
    fake_prob = float(np.mean(bundle["model"].predict_proba(X)[:, 1]))
    return {
        "model_name": "Voice-conversion detector (custom, local)",
        "model_id": CUSTOM_MODEL_ID,
        "is_fake": fake_prob > 0.5,
        "confidence": fake_prob if fake_prob > 0.5 else 1 - fake_prob,
        "fake_prob": fake_prob,
    }


def _fake_label_index(id2label: dict) -> int:
    for idx, label in id2label.items():
        if any(word in label.lower() for word in ("fake", "spoof", "synthetic", "ai")):
            return int(idx)
    return 1


def decode_audio(audio_bytes: bytes, suffix: str = ".wav"):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
    finally:
        os.unlink(tmp_path)
    return audio, sr


def run_model(audio: np.ndarray, model_name: str) -> dict:
    model_id = MODELS[model_name]
    if model_id == CUSTOM_MODEL_ID:
        return _run_custom_model(audio)
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


def compute_timeline(
    audio: np.ndarray,
    sr: int,
    model_name: str,
    window_s: float = 2.0,
    hop_s: float = 1.0,
    max_windows: int = 24,
) -> list[dict]:
    duration = len(audio) / sr
    if duration < window_s * 1.2:
        return []

    window = int(window_s * sr)
    hop = int(hop_s * sr)
    span = len(audio) - window
    n_windows = span // hop + 1
    if n_windows > max_windows:
        hop = max(1, span // (max_windows - 1))

    points = []
    start = 0
    while start + window <= len(audio):
        segment = audio[start:start + window]
        result = run_model(segment, model_name)
        points.append({"time": (start + window / 2) / sr, "fake_prob": result["fake_prob"]})
        start += hop
    return points


def compute_spectrogram(audio: np.ndarray, sr: int, n_mels: int = 80, max_frames: int = 240) -> dict:
    """A mel-spectrogram in dB, downsampled to a JSON-friendly size.

    n_mels=80 / max_frames=240 keeps the payload to roughly 80x240 floats
    (~19k numbers) regardless of clip length, instead of scaling with
    duration — a 30s clip and a 3s clip cost about the same to send.
    """
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=sr / 2)
    db = librosa.power_to_db(mel, ref=np.max)

    if db.shape[1] > max_frames:
        idx = np.linspace(0, db.shape[1] - 1, max_frames).astype(int)
        db = db[:, idx]

    return {
        "z": np.round(db, 1).tolist(),  # [n_mels][frames], dB relative to clip peak
        "duration": len(audio) / sr,
        "n_mels": n_mels,
    }


def analyze(
    audio_bytes: bytes,
    model_names: list[str],
    suffix: str = ".wav",
    include_extras: bool = True,
) -> dict:
    """include_extras=False skips the timeline/waveform/spectrogram — used
    by the live-recording ticker, which calls this every ~1.5s and only
    needs the bare fake-probability number back as fast as possible.
    """
    audio, sr = decode_audio(audio_bytes, suffix=suffix)
    results = [run_model(audio, name) for name in model_names]
    duration = len(audio) / sr

    if not include_extras:
        return {"results": results, "duration": duration}

    timeline = compute_timeline(audio, sr, model_names[0])

    step = max(1, len(audio) // 2000)
    waveform_y = audio[::step].tolist()
    waveform_x = (np.arange(len(waveform_y)) * step / sr).tolist()

    return {
        "results": results,
        "duration": duration,
        "timeline": timeline,
        "timeline_model": model_names[0],
        "waveform": {"x": waveform_x, "y": waveform_y},
        "spectrogram": compute_spectrogram(audio, sr),
    }
