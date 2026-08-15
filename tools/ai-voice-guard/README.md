# AI Voice Guard

Checks whether an audio clip sounds like a real human voice or an
AI-generated / voice-cloned one, using a `wav2vec2` model fine-tuned on
ElevenLabs, Amazon Polly, and Speechify samples.

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

## Running it

```powershell
.venv\Scripts\python -m streamlit run app.py
```

This starts a local web server and should open your browser automatically
to `http://localhost:8501`. If it doesn't, open that URL yourself. Leave
the terminal window open while you use the app — closing it stops the
server. Press `Ctrl+C` in that terminal to shut it down when you're done.

The first time you run it, it downloads the detection model from Hugging
Face (a few hundred MB), so the first launch will pause on "Loading
model…" for a bit. After that it's cached locally and starts instantly.

## Using the app

There are two tabs:

- **Upload** — drag in or browse to a WAV, MP3, FLAC, M4A, or OGG file.
  Analysis starts automatically once a file is selected.
- **Record** — click the record button, speak for a few seconds, click
  stop. Analysis starts automatically once you stop recording.

Either way, you'll get:

- A verdict card: **SOUNDS HUMAN** or **SOUNDS AI-GENERATED**, with a
  confidence percentage.
- A gauge showing the raw fake-voice probability (0–100%).
- A waveform view of the clip you submitted.

The sidebar keeps a running history of everything scanned in the current
browser session (cleared if you refresh the page or restart the server).

## Reading the result

Treat the verdict as a signal, not a certainty — it's a decent classifier,
not a guarantee. A clip scoring close to 50% is genuinely ambiguous to the
model; don't read too much into small differences near that line. For
anything where the answer actually matters (verifying a call, a claim, a
piece of evidence), corroborate with another method rather than relying on
this alone.

## Troubleshooting

- **App crashes immediately / `OSError` mentioning a `.dll` on Windows** —
  missing Visual C++ Redistributable. Install it from the link above and
  try again.
- **"Loading model…" hangs for a long time** — first run only; it's
  downloading the model. Needs an internet connection the first time.
  Subsequent runs use the local cache.
- **Port already in use** — another Streamlit app (or a previous run of
  this one) is still running on port 8501. Close it, or run with
  `--server.port 8502` to use a different port.
