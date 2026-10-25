# AI Voice Guard

Checks whether an audio clip sounds like a real human voice or an
AI-generated / voice-cloned one. Runs one or more independent `wav2vec2`
classifiers (each fine-tuned separately for this) and shows either a single
verdict or a consensus across several models. Sits behind a login page.

**Architecture:** a small Python backend (FastAPI) does the actual ML
inference and exposes it as a JSON API; the UI is a plain HTML/CSS/
JavaScript frontend (no framework, no build step) served by that same
backend. There's no separate frontend server to run — one process serves
both the page and the API.

```
tools/ai-voice-guard/
  backend/
    main.py        FastAPI app: routes, static file serving
    auth.py         login/session logic
    inference.py    model loading, scoring, the suspicion timeline
  static/
    index.html
    style.css
    app.js           all frontend logic — auth, upload/record, charts
  requirements.txt
  .env.example        template — copy to .env and fill in your own values
```

## Setup (first time only)

Requires Python 3.12 and, on Windows, the
[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
(PyTorch needs it — without it the app crashes on startup with a DLL error).

```powershell
cd "ai-voice-guard"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

On macOS/Linux, use `.venv/bin/pip` instead.

## Login credentials (required before the app will even start)

There is no hardcoded or auto-generated fallback credential anywhere in
this version — the app deliberately **refuses to start** if login isn't
configured, rather than falling back to anything guessable.

1. Copy `.env.example` to `.env` in this same folder.
2. Edit `.env` and set `APP_USERNAME` and `APP_PASSWORD` to whatever you
   want.

`.env` is listed in the repo's `.gitignore` — it is never committed, never
pushed, never visible on GitHub. It's a plain local file that only exists
on whichever machine you put it on; deploying this elsewhere (Render, a
VPS, etc.) means setting `APP_USERNAME`/`APP_PASSWORD` as that platform's
own environment variables instead of shipping a `.env` file with the code.

If you'd rather not have the plaintext password sitting in `.env` at all,
set `APP_PASSWORD_HASH` instead (a sha256 hex digest) — see the comment in
`.env.example` for the one-liner that generates it.

Be clear-eyed about what this login actually is: a lightweight gate to
keep casual visitors out, backed by an in-memory session cookie. It is
**not** production-grade auth — no rate limiting, no per-user accounts, no
audit log, and sessions reset if the server process restarts. Don't rely
on it to protect anything sensitive.

## Running it

```powershell
.venv\Scripts\python -m uvicorn backend.main:app --reload
```

(`--reload` restarts the server automatically when you edit a `.py` file —
drop it for a production-style run.) Then open **http://localhost:8000**
in your browser. Leave the terminal window open while you use the app;
`Ctrl+C` there shuts it down.

The first time you select a given model, it downloads from Hugging Face
(roughly 100–400MB) and that first analysis will be slow. After that it's
cached locally (via the standard Hugging Face cache) and loads fast for
the rest of that server process's lifetime.

## Using the app

Log in first (see above). In the sidebar, pick **one or more models**
under "Models to run":

- **wav2vec2-XLSR (deepfake voice)** — the original/default model.
- **wav2vec2 (deepfake audio)**
- **wav2vec2-XLSR-large (deepfake audio)**
- **wav2vec2 (deepfake audio v2)**
- **Voice-conversion detector (custom, local)** — see below.

With one model selected, you get a single big verdict card (**SOUNDS
HUMAN** / **SOUNDS AI-GENERATED**), a gauge showing the raw fake-voice
probability, and a waveform view.

With more than one model selected, you instead get a consensus banner
("3 of 4 models flag AI-generated · average fake-voice probability 81%")
plus a small card per model showing its individual verdict and confidence
— useful for seeing whether the models agree, rather than trusting a
single classifier blindly. Changing which models are selected re-runs the
analysis automatically on whatever clip is currently loaded.

The overall verdict is **any model flags it → flagged**, not a majority.
Testing against a labeled deepfake set showed why: these four models were
evidently trained on different fake styles, and some are reliably blind to
styles the others catch. Requiring a majority meant a model's correct catch
could be silently overruled by others that never would have caught it
either way — measured consensus recall on that test set was 0% under
majority vote vs. ~46% under any-model-flags, for a modest rise in false
positives. Treat the per-model breakdown as the real signal; the banner is
just "did anything go off," not a vote tally to trust blindly.

There are two ways to submit audio:

- **Upload** — drag a file onto the dropzone, or click it to browse.
  Accepts WAV, MP3, FLAC, M4A, or OGG.
- **Record** — click to start recording from your microphone (the browser
  will ask for permission). While you're recording, a **live gauge**
  updates every ~1.5 seconds with a real-time fake-probability reading
  from whatever's been captured so far (scored with just your first
  selected model, to keep it fast) — a live preview, not the final
  answer. Click the button again to stop; the complete recording is then
  run through the full analysis below, exactly like an uploaded file.

Either way, analysis starts automatically. Results include:

- A **suspicion timeline** for clips longer than about 2.5 seconds — the
  clip is scored in overlapping 2-second windows so you can see *where*
  in the recording it looks most synthetic, instead of one number
  averaged across the whole thing (useful for spotting a spliced-in or
  partially-cloned segment in an otherwise real recording).
- A **spectrogram** — a visual frequency-over-time view of the clip
  (computed as a mel-spectrogram), rendered as a themed heatmap. Useful
  for spotting things a plain waveform hides: unnatural frequency bands,
  cut/splice points, or the kind of artifacts synthetic speech sometimes
  leaves behind.

You can download a JSON report of the full result (verdicts + suspicion
timeline; the spectrogram itself isn't included, it's sized for display
rather than a report) via the button below the charts. The sidebar keeps
a running history of everything scanned in the current browser tab
(persisted across a page refresh via `sessionStorage`, cleared when the
tab closes).

## Reading the result

Treat any verdict as a signal, not a certainty — these are decent
classifiers, not guarantees, and different models can and do disagree on
the same clip (that disagreement is informative — if it's a coin flip
across models, don't trust it). A clip scoring close to 50% is genuinely
ambiguous; don't read too much into small differences near that line. For
anything where the answer actually matters (verifying a call, a claim, a
piece of evidence), corroborate with another method rather than relying on
this alone.

**Known blind spot:** all four wav2vec2 models struggle specifically with
voice-conversion-style fakes (the same speaker's real words/cadence with
just the timbre swapped onto another identity — e.g. RVC/so-vits output),
as opposed to TTS/vocoder-style synthesis. Tested against a labeled
Kaggle set of voice-conversion clips, the best of the four models still
only caught ~46% of known fakes; two of the four essentially never flagged
one (near-zero fake-probability on every fake sample tested, not just
uncertain). If you specifically need to catch cloned-voice calls or
conversion-style spoofing, don't treat a "SOUNDS HUMAN" verdict here as
reassurance — corroborate some other way. See
`tools/ai-voice-guard/test-samples/` for the test harness and full
results if you want to reproduce or extend this.

### The "Voice-conversion detector (custom, local)" model

Built specifically to cover the blind spot above. It's not a transformer —
there's no GPU on the reference dev machine, and full fine-tuning of a
wav2vec2 model on CPU wasn't practical. Instead it's a **GradientBoosting
classifier over classical audio features** (chroma, RMS, spectral
centroid/bandwidth/rolloff, zero-crossing rate, 20 MFCCs — averaged per
1-second window, clip score = mean probability across windows), trained on
the same Kaggle voice-conversion dataset used for testing above.

**Methodology, so the numbers below can be trusted:** the dataset has 8
speaker identities, each voice-converted to/from every other identity (56
clips) plus 8 original "real" recordings. 2 of the 8 identities (`ryan`,
`taylor`) were held out of training *entirely* — not just their files, but
every fake clip involving them as either source or target — so evaluation
measures generalization to voices the model never saw, not memorization of
a specific recording's background noise or channel characteristics. Model
selection (GradientBoosting over RandomForest/LogisticRegression) was by
F1 score with a hard reject on >15% false-positive rate, specifically
because an earlier pass on raw recall alone selected a model that flagged
52% of real audio as fake — useless for a tool meant to be trusted.

**Held-out-identity results:** 75.5% recall on fakes it never saw in
training, 99.8% precision, 2.6% false-positive rate on real audio. That's
a meaningful jump over the best wav2vec2 model's ~46% recall on the same
attack style, with far fewer false alarms.

**Scope limitation — read this before trusting it broadly:** this model
is trained and validated *only* on this one Kaggle voice-conversion
dataset — 8 English-speaking public figures, one conversion technique,
studio-ish audio quality. It has not been tested against TTS-based fakes,
other conversion techniques, other languages, phone-quality audio, or
adversarial/noisy conditions. Treat it as a specialist for exactly the
attack style it was trained on, not a general-purpose replacement for the
other four models — run it *alongside* them, not instead of them.
Training script, full metrics, and the exact train/test split are in
`tools/ai-voice-guard/test-samples/train_classifier.py` and
`train_results.json`.

## Troubleshooting

- **App refuses to start, error mentions `APP_USERNAME`/`APP_PASSWORD`** —
  you haven't created `.env` yet, or it's missing a value. See "Login
  credentials" above.
- **`OSError` mentioning a `.dll` on Windows** — missing Visual C++
  Redistributable. Install it from the link above and try again.
- **First analysis with a given model is slow** — it's downloading from
  Hugging Face the first time. Needs an internet connection then;
  subsequent runs use the local cache.
- **Microphone recording doesn't work** — the browser needs an explicit
  permission grant for the mic, and (outside of `localhost`) most browsers
  require HTTPS for `getUserMedia` to work at all.
- **Port already in use** — another process is already on 8000. Close it,
  or run with `--port 8001` (and open that port in your browser instead).
- **Selecting multiple models is slow / uses a lot of memory** — each
  model is a separate multi-hundred-MB neural net loaded into RAM at the
  same time. On a resource-constrained host, stick to one or two models
  at once.
