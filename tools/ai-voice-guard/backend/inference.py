"""ML inference: model loading, per-clip and per-window scoring."""

import os
import tempfile

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODELS = {
    "wav2vec2-XLSR (deepfake voice)": "garystafford/wav2vec2-deepfake-voice-detector",
    "wav2vec2 (deepfake audio)": "mo-thecreator/Deepfake-audio-detection",
    "wav2vec2-XLSR-large (deepfake audio)": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
    "wav2vec2 (deepfake audio v2)": "MelodyMachine/Deepfake-audio-detection-V2",
}
DEFAULT_MODEL = "wav2vec2-XLSR (deepfake voice)"

_model_cache: dict[str, tuple] = {}


def load_model(model_id: str):
    if model_id not in _model_cache:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        model.eval()
        _model_cache[model_id] = (feature_extractor, model)
    return _model_cache[model_id]


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
