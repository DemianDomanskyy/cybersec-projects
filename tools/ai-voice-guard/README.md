# AI Voice Guard

Checks whether an audio clip sounds like a real human voice or an
AI-generated / voice-cloned one. Runs one or more independent `wav2vec2`
classifiers (each fine-tuned separately for this) and shows either a single
verdict or a consensus across several models. Sits behind a login page.

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

## Login

The app is gated behind a username/password screen. There is no hardcoded
default credential anywhere in the source — nothing to leak, nothing to
remove from git history later. Instead:

- If you haven't set `APP_USERNAME`/`APP_PASSWORD`, the app generates a
  random password each time the server process starts and prints it to
  the **server's own console log** (not the web page). Username defaults
  to `admin`. Check your terminal for a line like:
  `[AI Voice Guard] No APP_PASSWORD configured — generated a one-time
  local login password: <random string>`. Use that to log in for this run;
  it changes the next time you restart the server.
- For a stable login (recommended once you're not just poking at it
  locally), set:
  - `APP_USERNAME` — your chosen username.
  - `APP_PASSWORD` — your chosen password (hashed in memory before
    comparison, but it's still simplest to set it as plain text here).

  Or set `APP_PASSWORD_HASH` instead of `APP_PASSWORD` if you'd rather not
  have the plaintext password sitting in your environment at all. Generate
  it with:

  ```powershell
  python -c "import hashlib; print(hashlib.sha256('yourpassword'.encode()).hexdigest())"
  ```

Be clear-eyed about what this actually is: a lightweight gate to keep
casual visitors out of a demo, backed by Streamlit's session state. It is
**not** real access control — no rate limiting, no per-user accounts, no
audit log. Don't rely on it to protect anything sensitive.

## Running it

```powershell
.venv\Scripts\python -m streamlit run app.py
```

This starts a local web server and should open your browser automatically
to `http://localhost:8501`. If it doesn't, open that URL yourself. Leave
the terminal window open while you use the app — closing it stops the
server. Press `Ctrl+C` in that terminal to shut it down when you're done.

The first time you run it, each model you select downloads from Hugging
Face (roughly 100–400MB apiece), so the first analysis using a given model
will pause on "Loading model(s)…" for a bit. After that it's cached
locally and starts instantly.

## Using the app

Log in first (see above). Then, in the sidebar, pick **one or more
models** under "Models to run":

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
— useful when you want to see whether the models agree or not, rather than
trusting a single classifier blindly.

There are two ways to submit audio:

- **Upload** — drag in or browse to a WAV, MP3, FLAC, M4A, or OGG file.
- **Record** — click the record button, speak for a few seconds, click
  stop.

Either way, analysis starts automatically, and you can download a JSON
report of the result via the button below the waveform. The sidebar keeps
a running history of everything scanned in the current browser session
(cleared if you refresh the page or restart the server).

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

- **App crashes immediately / `OSError` mentioning a `.dll` on Windows** —
  missing Visual C++ Redistributable. Install it from the link above and
  try again.
- **"Loading model(s)…" hangs for a long time** — first run only per
  model; it's downloading from Hugging Face. Needs an internet connection
  the first time. Subsequent runs use the local cache.
- **Forgot the login password** — check whatever `APP_USERNAME` /
  `APP_PASSWORD` (or `APP_PASSWORD_HASH`) you set in your environment; if
  none are set, check the server's console log for the generated
  one-time password (see "Login" above) — restarting the server generates
  a new one.
- **Port already in use** — another Streamlit app (or a previous run of
  this one) is still running on port 8501. Close it, or run with
  `--server.port 8502` to use a different port.
- **Selecting multiple models is slow / uses a lot of memory** — each
  model is a separate multi-hundred-MB neural net loaded into RAM at the
  same time. On a resource-constrained host (see the Render deploy notes
  in the repo's `render.yaml`), stick to one or two models at once.
