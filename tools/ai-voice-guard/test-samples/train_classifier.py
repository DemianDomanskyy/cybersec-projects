"""Trains a classical ML classifier (RandomForest) to detect voice-conversion
style deepfakes, specifically targeting the blind spot the 4 wav2vec2 models
showed on the Kaggle DEEP-VOICE set (see test-samples/results.json).

No GPU is available on this machine, so full gradient fine-tuning of a
transformer is impractical (hours-to-days on CPU, likely to overfit on this
little data anyway). Classical audio features (chroma/RMS/spectral/MFCC,
same style as the dataset's own DATASET-balanced.csv) + a RandomForest train
in seconds-to-minutes on CPU and are a legitimate, well-established approach
for this kind of audio classification task.

Critically: this script extracts features itself (rather than reusing the
provided DATASET-balanced.csv) because that CSV has no file/identity
column, so there is no way to know which rows came from which recording.
A random split over it risks segments from the same recording landing in
both train and test, which inflates accuracy by letting the model key off
recording-specific channel/noise characteristics instead of real
fake-detection signal. Here, 2 of the 8 identities (ryan, taylor) are held
out of training ENTIRELY and used only for test, so the reported accuracy
reflects generalization to unseen speakers.

Usage:
    .venv\\Scripts\\python test-samples\\train_classifier.py
"""

import glob
import json
import os
import time

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted", "KAGGLE", "AUDIO")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "custom_model")

TEST_IDENTITIES = {"ryan", "taylor"}  # held out of training entirely
WINDOW_S = 1.0
HOP_S = 1.0
MAX_SECONDS_PER_FILE = 180  # cap load/feature time; ~180 segments/file max

FEATURE_NAMES = [
    "chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff",
    "zero_crossing_rate",
] + [f"mfcc{i}" for i in range(1, 21)]


def extract_features(y, sr):
    """One feature row per WINDOW_S window, matching the Kaggle CSV's columns."""
    window = int(WINDOW_S * sr)
    hop = int(HOP_S * sr)
    rows = []
    start = 0
    while start + window <= len(y):
        seg = y[start:start + window]
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


def identity_of(filename, is_real):
    base = os.path.basename(filename).replace(".wav", "")
    if is_real:
        return base.replace("-original", "").lower()
    src, tgt = base.split("-to-")
    return src.lower(), tgt.lower()


def gather_files():
    real_files = sorted(glob.glob(os.path.join(DATA_DIR, "REAL", "*.wav")))
    fake_files = sorted(glob.glob(os.path.join(DATA_DIR, "FAKE", "*.wav")))

    train_real, test_real, train_fake, test_fake = [], [], [], []
    for f in real_files:
        ident = identity_of(f, True)
        (test_real if ident in TEST_IDENTITIES else train_real).append(f)
    for f in fake_files:
        src, tgt = identity_of(f, False)
        if src in TEST_IDENTITIES or tgt in TEST_IDENTITIES:
            test_fake.append(f)
        else:
            train_fake.append(f)
    return train_real, train_fake, test_real, test_fake


def build_split(real_files, fake_files, label_tag):
    X, y, groups = [], [], []
    t0 = time.time()
    for f in real_files:
        audio, sr = librosa.load(f, sr=16000, mono=True, duration=MAX_SECONDS_PER_FILE)
        rows = extract_features(audio, sr)
        X.extend(rows)
        y.extend([0] * len(rows))  # 0 = real
        groups.extend([os.path.basename(f)] * len(rows))
        print(f"  [{label_tag}] REAL {os.path.basename(f)}: {len(rows)} segments [{time.time()-t0:.0f}s elapsed]")
    for f in fake_files:
        audio, sr = librosa.load(f, sr=16000, mono=True, duration=MAX_SECONDS_PER_FILE)
        rows = extract_features(audio, sr)
        X.extend(rows)
        y.extend([1] * len(rows))  # 1 = fake
        groups.extend([os.path.basename(f)] * len(rows))
        print(f"  [{label_tag}] FAKE {os.path.basename(f)}: {len(rows)} segments [{time.time()-t0:.0f}s elapsed]")
    return np.array(X), np.array(y), groups


def balance(X, y, seed=0):
    rng = np.random.RandomState(seed)
    idx_real = np.where(y == 0)[0]
    idx_fake = np.where(y == 1)[0]
    n = min(len(idx_real), len(idx_fake))
    idx_real = rng.choice(idx_real, n, replace=False)
    idx_fake = rng.choice(idx_fake, n, replace=False)
    idx = np.concatenate([idx_real, idx_fake])
    rng.shuffle(idx)
    return X[idx], y[idx]


def main():
    train_real, train_fake, test_real, test_fake = gather_files()
    print(f"Train identities: REAL={len(train_real)} files, FAKE pairs={len(train_fake)} files")
    print(f"Test (held-out) identities: REAL={len(test_real)} files, FAKE pairs={len(test_fake)} files")
    print(f"Held-out test identities: {sorted(TEST_IDENTITIES)}\n")

    print("Extracting TRAIN features...")
    X_train, y_train, _ = build_split(train_real, train_fake, "train")
    print(f"\nExtracting TEST features...")
    X_test, y_test, test_groups = build_split(test_real, test_fake, "test")

    print(f"\nRaw train rows: {len(y_train)} (real={sum(y_train==0)}, fake={sum(y_train==1)})")
    print(f"Raw test rows: {len(y_test)} (real={sum(y_test==0)}, fake={sum(y_test==1)})")

    X_train_bal, y_train_bal = balance(X_train, y_train)
    print(f"Balanced train rows: {len(y_train_bal)}")

    scaler = StandardScaler().fit(X_train_bal)
    X_train_s = scaler.transform(X_train_bal)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=0, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    }

    results = {}
    # Select by F1 (balances recall and precision) rather than recall alone --
    # a model that "catches" fakes by flagging everything (high recall, poor
    # precision) is worse than useless for a security tool: it trains users
    # to ignore the warning. Reject any candidate whose false-positive rate
    # on real audio is too high to be usable, even if its recall is good.
    MAX_FP_RATE = 0.15  # a model wrongly flagging >15% of real clips is unusable
    best_name, best_model, best_f1 = None, None, -1
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_train_s, y_train_bal)
        pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, pred)
        rec = recall_score(y_test, pred)  # recall on FAKE (class 1)
        prec = precision_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        cm = confusion_matrix(y_test, pred).tolist()
        tn, fp = cm[0]
        fp_rate = fp / (tn + fp) if (tn + fp) else 0
        results[name] = {"accuracy": acc, "recall_fake": rec, "precision_fake": prec, "f1_fake": f1,
                          "fp_rate_on_real": fp_rate, "confusion_matrix": cm}
        print(f"\n{name} (trained in {time.time()-t0:.1f}s, segment-level eval on held-out identities {sorted(TEST_IDENTITIES)}):")
        print(f"  accuracy={acc:.1%}  recall(fake)={rec:.1%}  precision(fake)={prec:.1%}  f1={f1:.1%}  fp_rate_on_real={fp_rate:.1%}  confusion={cm}")
        if fp_rate > MAX_FP_RATE:
            print(f"  -> rejected: false-positive rate {fp_rate:.1%} exceeds {MAX_FP_RATE:.0%} threshold, too noisy to ship")
            continue
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, clf, f1

    if best_model is None:
        raise SystemExit("No candidate model met the false-positive-rate bar -- not saving anything.")
    print(f"\nBest model by F1 (among candidates under {MAX_FP_RATE:.0%} FP rate): {best_name}")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump({"model": best_model, "scaler": scaler, "feature_names": FEATURE_NAMES, "model_name": best_name},
                os.path.join(OUT_DIR, "voice_conversion_classifier.joblib"))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_results.json"), "w") as fh:
        json.dump({
            "test_identities_held_out": sorted(TEST_IDENTITIES),
            "window_s": WINDOW_S,
            "max_seconds_per_file": MAX_SECONDS_PER_FILE,
            "train_rows_balanced": len(y_train_bal),
            "test_rows": len(y_test),
            "results": results,
            "best_model": best_name,
        }, fh, indent=2)

    print(f"\nSaved classifier to {os.path.join(OUT_DIR, 'voice_conversion_classifier.joblib')}")
    print(f"Saved metrics to test-samples/train_results.json")


if __name__ == "__main__":
    main()
