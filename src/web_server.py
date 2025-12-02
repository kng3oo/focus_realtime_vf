import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .camera_worker import (
    process_frame,
    get_live_state,
    get_session_meta,
    save_snapshot,
    start_calibration,
    log_manager,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Focus Realtime Dashboard")

# CORS 허용 (모바일, PC 모두 접속 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static/* 서빙
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------
# 기본 페이지
# --------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# --------------------------------------------------------------------
# Live State / Session API
# --------------------------------------------------------------------
@app.get("/api/live")
def api_live():
    return get_live_state()


@app.get("/api/session")
def api_session():
    return get_session_meta()


# --------------------------------------------------------------------
# 로그 다운로드
# --------------------------------------------------------------------
@app.get("/api/download_log")
def api_download_log():
    path = log_manager.get_csv_path()
    return FileResponse(
        path,
        filename=os.path.basename(path),
        media_type="text/csv",
    )


# --------------------------------------------------------------------
# Snapshot 저장
# --------------------------------------------------------------------
@app.post("/api/snapshot")
async def api_snapshot(payload: dict):
    user = payload.get("user")
    ok = save_snapshot(user=user)
    return {"ok": ok}


# --------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------
@app.post("/api/calibrate/start")
def api_calibrate_start():
    ok = start_calibration()
    return {"ok": ok}

