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

With one model selected, you get a single big verdict card (**SOUNDS
HUMAN** / **SOUNDS AI-GENERATED**), a gauge showing the raw fake-voice
probability, and a waveform view.

With more than one model selected, you instead get a consensus banner
("3 of 4 models say AI-generated · average fake-voice probability 81%")
plus a small card per model showing its individual verdict and confidence
— useful for seeing whether the models agree, rather than trusting a
single classifier blindly. Changing which models are selected re-runs the
analysis automatically on whatever clip is currently loaded.

There are two ways to submit audio:

- **Upload** — drag a file onto the dropzone, or click it to browse.
  Accepts WAV, MP3, FLAC, M4A, or OGG.
- **Record** — click to start recording from your microphone (the browser
  will ask for permission), click again to stop.

Either way, analysis starts automatically. For clips longer than about
2.5 seconds, you'll also get a **suspicion timeline** — the clip is
scored in overlapping 2-second windows so you can see *where* in the
recording it looks most synthetic, instead of one number averaged across
the whole thing (useful for spotting a spliced-in or partially-cloned
segment in an otherwise real recording). You can download a JSON report
of the full result, timeline included, via the button below. The sidebar
keeps a running history of everything scanned in the current browser tab
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
