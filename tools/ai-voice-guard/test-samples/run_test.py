"""Scores the Kaggle DEEP-VOICE REAL/FAKE sample set against every model in
backend.inference.MODELS and reports per-model + consensus accuracy.

Each clip is truncated to CLIP_SECONDS (default 30s) before scoring -- the
source clips are ~10 minutes each, and wav2vec2's attention doesn't scale
to that length anyway; a 30s window is representative of how the app is
actually used.

Usage (from tools/ai-voice-guard, with the venv active):
    .venv\\Scripts\\python test-samples\\run_test.py
"""

import argparse
import glob
import json
import os
import sys
import time

# Make backend/ importable when run from tools/ai-voice-guard
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa

from backend.inference import MODELS, run_model

CLIP_SECONDS = 30
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted", "KAGGLE", "AUDIO")


def load_samples(limit_per_class=None):
    samples = []
    for label, subdir in (("REAL", "REAL"), ("FAKE", "FAKE")):
        paths = sorted(glob.glob(os.path.join(DATA_DIR, subdir, "*.wav")))
        if limit_per_class:
            paths = paths[:limit_per_class]
        for p in paths:
            samples.append((p, label))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-class", type=int, default=None,
                         help="only test the first N files per REAL/FAKE class")
    parser.add_argument("--clip-seconds", type=float, default=CLIP_SECONDS)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"))
    args = parser.parse_args()

    samples = load_samples(args.limit_per_class)
    print(f"Testing {len(samples)} clips ({sum(1 for _, l in samples if l == 'REAL')} REAL / "
          f"{sum(1 for _, l in samples if l == 'FAKE')} FAKE), {args.clip_seconds}s window each.\n")

    model_names = list(MODELS.keys())
    # per-model counters: [tp, tn, fp, fn] where "positive" = FAKE
    stats = {m: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for m in model_names}
    consensus_stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    rows = []

    t_start = time.time()
    for i, (path, true_label) in enumerate(samples, 1):
        fname = os.path.basename(path)
        print(f"[{i}/{len(samples)}] {true_label:4s}  {fname} ...", end=" ", flush=True)
        t0 = time.time()
        audio, sr = librosa.load(path, sr=16000, mono=True, duration=args.clip_seconds)

        per_model = {}
        votes_fake = 0
        for m in model_names:
            try:
                result = run_model(audio, m)
            except Exception as e:
                print(f"\n  ERROR running {m} on {fname}: {e}")
                continue
            pred_label = "FAKE" if result["is_fake"] else "REAL"
            per_model[m] = {"pred": pred_label, "fake_prob": round(result["fake_prob"], 4)}
            votes_fake += 1 if result["is_fake"] else 0

            s = stats[m]
            if true_label == "FAKE" and pred_label == "FAKE":
                s["tp"] += 1
            elif true_label == "REAL" and pred_label == "REAL":
                s["tn"] += 1
            elif true_label == "REAL" and pred_label == "FAKE":
                s["fp"] += 1
            else:
                s["fn"] += 1

        consensus_pred = "FAKE" if votes_fake > len(model_names) / 2 else "REAL"
        if true_label == "FAKE" and consensus_pred == "FAKE":
            consensus_stats["tp"] += 1
        elif true_label == "REAL" and consensus_pred == "REAL":
            consensus_stats["tn"] += 1
        elif true_label == "REAL" and consensus_pred == "FAKE":
            consensus_stats["fp"] += 1
        else:
            consensus_stats["fn"] += 1

        rows.append({"file": fname, "true_label": true_label, "per_model": per_model,
                      "consensus_pred": consensus_pred, "votes_fake": votes_fake})
        print(f"consensus={consensus_pred} ({votes_fake}/{len(model_names)} fake) [{time.time()-t0:.1f}s]")

    def report(name, s):
        total = s["tp"] + s["tn"] + s["fp"] + s["fn"]
        acc = (s["tp"] + s["tn"]) / total if total else 0
        print(f"\n{name}")
        print(f"  accuracy: {acc:.1%}  (TP={s['tp']} TN={s['tn']} FP={s['fp']} FN={s['fn']})")
        return acc

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    summary = {}
    for m in model_names:
        summary[m] = report(m, stats[m])
    summary["CONSENSUS"] = report("CONSENSUS (majority vote)", consensus_stats)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")

    with open(args.out, "w") as f:
        json.dump({"clip_seconds": args.clip_seconds, "summary_accuracy": summary,
                    "per_model_confusion": stats, "consensus_confusion": consensus_stats,
                    "rows": rows}, f, indent=2)
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
