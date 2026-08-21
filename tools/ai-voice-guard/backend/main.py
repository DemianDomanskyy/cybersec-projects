"""AI Voice Guard API — FastAPI backend serving the JS frontend and the
audio-analysis endpoints. Run with: uvicorn backend.main:app
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads a local, gitignored .env file if present

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, inference

if not auth.credentials_configured():
    raise RuntimeError(
        "APP_USERNAME and (APP_PASSWORD or APP_PASSWORD_HASH) must be set — "
        "copy .env.example to .env in tools/ai-voice-guard/ and fill in your "
        "own values. There is no built-in default login."
    )

app = FastAPI(title="AI Voice Guard API")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/login")
def login(body: LoginRequest, response: Response):
    if not auth.verify_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Wrong username or password")
    token = auth.create_session()
    response.set_cookie(
        key=auth.SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 12,  # 12 hours
    )
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response, session: str = Depends(auth.require_session)):
    auth.destroy_session(session)
    response.delete_cookie(auth.SESSION_COOKIE_NAME)
    return {"ok": True}


@app.get("/api/me")
def me(session: str = Depends(auth.require_session)):
    return {"ok": True}


@app.get("/api/models")
def list_models(session: str = Depends(auth.require_session)):
    return {"models": list(inference.MODELS.keys()), "default": inference.DEFAULT_MODEL}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    models: str = Form(...),
    extras: bool = Form(True),
    session: str = Depends(auth.require_session),
):
    model_names = json.loads(models)
    if not model_names:
        raise HTTPException(status_code=400, detail="Select at least one model")
    for name in model_names:
        if name not in inference.MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {name}")

    audio_bytes = await file.read()
    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    try:
        result = inference.analyze(audio_bytes, model_names, suffix=suffix, include_extras=extras)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode this audio — try a WAV, MP3, FLAC, M4A, "
                f"or OGG file instead. ({exc})"
            ),
        ) from exc
    return result
