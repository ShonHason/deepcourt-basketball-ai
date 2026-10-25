"""
DeepCourt — unified server on port 8510.
Landing page, auth API, and Streamlit dashboard all live here.

Run:  python landing/api.py
Open: http://localhost:8510/
"""
import os, secrets, sys, urllib.parse, uuid as _uuid
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import bcrypt
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from src.database import save_user, load_user_by_email, load_user, patch_user, save_job, save_other_video

_PROJECT_ROOT = Path(__file__).parent.parent
_VIDEO_DIR    = _PROJECT_ROOT / "data" / "videos"

GOOGLE_CLIENT_ID     = os.environ["Client_ID"].strip()
GOOGLE_CLIENT_SECRET = os.environ["Client_Secret"].strip()
REDIRECT_URI         = "http://localhost:8510/api/auth/google/callback"
GOOGLE_AUTH_URL      = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL     = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL  = "https://www.googleapis.com/oauth2/v3/userinfo"

# In-memory store for pending Google sign-ins (consumed once)
pending_google: dict[str, dict] = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Email registration ────────────────────────────────────────────────────────

@app.post("/api/register")
def register(
    first_name:       str = Form(...),
    last_name:        str = Form(...),
    email:            str = Form(...),
    password:         str = Form(...),
    confirm_password: str = Form(...),
    birth_date:       str = Form(...),
    track:            str = Form(...),
    club:             str = Form(...),
    club_other:       str = Form(""),
    position:         str = Form(...),
    height:           str = Form(""),
    dominant_hand:    str = Form("right"),
    age_group:        str = Form(""),
):
    if password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if load_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    display_name = f"{first_name.strip()} {last_name.strip()}"
    club_name    = club_other.strip() if club == "other" else club
    height_cm    = int(height) if height and str(height).isdigit() else None
    pw_hash      = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    user_id = save_user(
        email=email,
        display_name=display_name,
        auth_provider="email",
        position=position,
        height_cm=height_cm,
        dominant_hand=dominant_hand or "right",
        birth_date=birth_date,
        competition_track=track,
        age_group=age_group,
        club_name=club_name,
        password_hash=pw_hash,
    )
    return JSONResponse({"success": True, "user_id": user_id, "name": display_name})


# ── Google registration ───────────────────────────────────────────────────────

@app.post("/api/register/google")
def register_google(
    google_sub:  str = Form(...),
    first_name:  str = Form(...),
    last_name:   str = Form(...),
    email:       str = Form(...),
    birth_date:  str = Form(...),
    track:       str = Form(...),
    club:        str = Form(...),
    club_other:  str = Form(""),
    position:    str = Form(...),
    height:      str = Form(""),
    dominant_hand: str = Form("right"),
    age_group:   str = Form(""),
):
    display_name = f"{first_name.strip()} {last_name.strip()}"
    club_name    = club_other.strip() if club == "other" else club
    height_cm    = int(height) if height and str(height).isdigit() else None

    if load_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user_id = save_user(
        email=email,
        display_name=display_name,
        auth_provider="google",
        position=position,
        height_cm=height_cm,
        dominant_hand=dominant_hand or "right",
        birth_date=birth_date,
        competition_track=track,
        age_group=age_group,
        club_name=club_name,
    )
    return JSONResponse({"success": True, "user_id": user_id, "name": display_name})


# ── Email login ───────────────────────────────────────────────────────────────

@app.post("/api/login")
def login(email: str = Form(...), password: str = Form(...)):
    user = load_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email")
    if user.get("auth_provider") == "google":
        raise HTTPException(status_code=400, detail="This account uses Google Sign In — please click 'Continue with Google'")
    pw_hash = user.get("password_hash") or ""
    if not pw_hash or not bcrypt.checkpw(password.encode(), pw_hash.encode()):
        raise HTTPException(status_code=401, detail="Incorrect password")
    return JSONResponse({
        "success":  True,
        "user_id":  user["id"],
        "name":     user.get("display_name", ""),
        "email":    user.get("email", ""),
        "position": user.get("position", ""),
        "club":     user.get("club_name", ""),
    })


# ── Google OAuth ──────────────────────────────────────────────────────────────

@app.get("/api/auth/google")
def google_login():
    params = (
        f"client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
        f"&access_type=offline"
        f"&prompt=select_account"
    )
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{params}")


@app.get("/api/auth/google/callback")
async def google_callback(code: str, intent: str = "login"):
    try:
        async with httpx.AsyncClient() as client:
            token_res = await client.post(GOOGLE_TOKEN_URL, data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  REDIRECT_URI,
                "grant_type":    "authorization_code",
            })
            token_data = token_res.json()
            if "error" in token_data:
                err = token_data.get("error_description", "OAuth error")
                return RedirectResponse(f"http://localhost:8510/login.html?error={err}")

            info_res  = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {token_data['access_token']}"}
            )
            user_info = info_res.json()

        email       = user_info.get("email", "")
        given_name  = user_info.get("given_name", "")
        family_name = user_info.get("family_name", "")

        if not email:
            return RedirectResponse("http://localhost:8510/login.html?error=no_email")

        existing = load_user_by_email(email)

        if intent == "login":
            if existing:
                uid        = existing["id"]
                name_param = (existing.get("display_name") or email).split()[0]
                return RedirectResponse(
                    f"http://localhost:8510/login.html?success=1&name={name_param}&user_id={uid}"
                )
            else:
                token = secrets.token_urlsafe(24)
                pending_google[token] = {
                    "email": email, "given_name": given_name,
                    "family_name": family_name, "google_sub": user_info.get("sub", ""),
                }
                return RedirectResponse(f"http://localhost:8510/register.html?google_token={token}")

        # REGISTER intent
        if existing:
            uid        = existing["id"]
            name_param = (existing.get("display_name") or email).split()[0]
            return RedirectResponse(
                f"http://localhost:8510/login.html?success=1&name={name_param}&user_id={uid}"
            )
        token = secrets.token_urlsafe(24)
        pending_google[token] = {
            "email": email, "given_name": given_name,
            "family_name": family_name, "google_sub": user_info.get("sub", ""),
        }
        return RedirectResponse(f"http://localhost:8510/register.html?google_token={token}")

    except Exception as exc:
        return RedirectResponse(
            f"http://localhost:8510/login.html?error={urllib.parse.quote(str(exc))}"
        )


@app.get("/api/user")
def get_user(user_id: str):
    user = load_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return JSONResponse({
        "user_id":  user["id"],
        "name":     user.get("display_name", ""),
        "club":     user.get("club_name", ""),
        "position": user.get("position", ""),
    })


@app.post("/api/user/{user_id}/update")
async def update_user_profile(
    user_id:    str,
    height_cm:  str = Form(""),
    club_name:  str = Form(""),
    club_other: str = Form(""),
    age_group:  str = Form(""),
    position:   str = Form(""),
    photo:      UploadFile = File(None),
):
    import base64
    try:
        height = int(float(height_cm.strip())) if height_cm.strip() else None
    except (ValueError, TypeError):
        height = None
    resolved_club = club_other.strip() if club_name.strip() == "__other__" else club_name.strip()
    kwargs: dict = {"club_name": resolved_club, "age_group": age_group.strip(), "position": position.strip()}
    if height is not None:
        kwargs["height_cm"] = height
    if photo and photo.filename:
        raw = await photo.read()
        if raw:
            mime = photo.content_type or "image/jpeg"
            kwargs["photo_b64"] = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    patch_user(user_id, **kwargs)
    return RedirectResponse(
        f"http://localhost:8510/dashboard?user_id={user_id}&page=profile",
        status_code=303,
    )


@app.get("/api/auth/google/pending")
def google_pending(token: str):
    data = pending_google.pop(token, None)
    if not data:
        raise HTTPException(status_code=404, detail="Token expired or not found")
    return JSONResponse(data)


# ── FAB video uploads (client-side fetch, no page navigation) ────────────────

@app.post("/api/upload/analyse")
async def api_upload_analyse(
    file:    UploadFile = File(...),
    user_id: str        = Form(...),
):
    _VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    fname = file.filename or "upload.mp4"
    ext   = Path(fname).suffix or ".mp4"
    jid   = _uuid.uuid4().hex
    path  = _VIDEO_DIR / f"{jid}{ext}"
    path.write_bytes(await file.read())
    job = {
        "id": jid, "name": fname, "path": str(path),
        "status": "queued", "progress": 0.0, "status_msg": "Queued",
        "result": None, "error": None, "user_id": user_id,
    }
    save_job(job)
    return JSONResponse({"success": True, "job_id": jid})


@app.post("/api/upload/other")
async def api_upload_other(
    file:    UploadFile = File(...),
    user_id: str        = Form(...),
    title:   str        = Form(""),
):
    other_dir = _VIDEO_DIR / "other"
    other_dir.mkdir(parents=True, exist_ok=True)
    vid_id    = _uuid.uuid4().hex
    fname     = file.filename or "upload.mp4"
    ext       = Path(fname).suffix or ".mp4"
    path      = other_dir / f"{vid_id}{ext}"
    path.write_bytes(await file.read())
    save_other_video(
        user_id=user_id,
        title=title.strip() or Path(fname).stem,
        file_path=str(path),
        video_id=vid_id,
    )
    return JSONResponse({"success": True, "video_id": vid_id})


# ── Dashboard — auth-gated iframe wrapper ─────────────────────────────────────

@app.get("/dashboard")
def dashboard_page(request: Request):
    user_id = request.query_params.get("user_id", "").strip()
    if not user_id:
        return RedirectResponse("http://localhost:8510/login.html")
    user = load_user(user_id)
    if not user:
        return RedirectResponse("http://localhost:8510/login.html?error=session_expired")

    qs  = str(request.query_params)
    src = f"http://localhost:8502/?{qs}"
    html = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>DeepCourt</title>"
        "<style>"
        "*{box-sizing:border-box;margin:0;padding:0}"
        "html,body{width:100%;height:100%;background:#054530;overflow:hidden}"
        "iframe{display:block;width:100%;height:100vh;border:none}"
        "</style></head><body>"
        f"<iframe src='{src}' allow='camera;microphone;clipboard-read;clipboard-write;fullscreen'></iframe>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


# ── Logout ───────────────────────────────────────────────────────────────────

@app.get("/logout")
def logout():
    html = (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
        "<script>"
        "localStorage.removeItem('dc_user');"
        "window.location.href = '/login.html';"
        "</script>"
        "</head><body></body></html>"
    )
    return HTMLResponse(content=html)


# ── Landing static files (must be last — catches everything else) ─────────────

landing_dir = os.path.abspath(os.path.dirname(__file__))
app.mount("/", StaticFiles(directory=landing_dir, html=True), name="landing")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8510, reload=False)
