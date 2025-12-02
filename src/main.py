# main.py
import os
import asyncio
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
import cv2

from .camera_worker import (
    process_frame, get_live_state, get_session_meta,
    save_snapshot, start_calibration,
    _user_slug, _user_run_dir, log_manager
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


# Thread pool for heavy frame processing
executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="Focus Realtime")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------------------------------------------------------
# index.html
# ----------------------------------------------------------
@app.get("/")
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


# ----------------------------------------------------------
# API: Live, Session
# ----------------------------------------------------------
@app.get("/api/live")
def api_live() -> Dict[str, Any]:
    return get_live_state()


@app.get("/api/session")
def api_session() -> Dict[str, Any]:
    return get_session_meta()


# ----------------------------------------------------------
# API: 로그 다운로드 (user_slug 기반)
# ----------------------------------------------------------
@app.get("/api/download_log")
def api_download_log():
    """
    사용자 폴더에 저장된 CSV 로그를 다운로드한다.
    """
    if not log_manager:
        return {"error": "log not ready"}

    path = log_manager.get_csv_path()

    return FileResponse(
        path,
        filename=os.path.basename(path),
        media_type="text/csv",
    )


# ----------------------------------------------------------
# API: 프레임 업로드 (user_slug 적용됨)
# ----------------------------------------------------------
@app.post("/api/frame")
async def api_frame(
    user: str = Form("", description="사용자 이름"),
    frame: UploadFile = File(...),
):
    """
    사용자 이름(user)에 따라 camera_worker 가:
      1) 사용자 폴더 생성 (runs/{slug}, screenshots/{slug})
      2) calib.json도 사용자 폴더에 저장
      3) 로그도 runs/{slug}/ 아래 기록
    """
    data = await frame.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is not None:
        # user 를 전달하면 camera_worker 내부에서 user_slug 기반 경로를 자동 반영
        process_frame(img, user=user or None)

    return {"ok": True, "user_slug": _user_slug}


# ----------------------------------------------------------
# API: 스냅샷 저장 (user_slug 기반)
# ----------------------------------------------------------
@app.post("/api/snapshot")
async def api_snapshot(payload: Optional[Dict[str, Any]] = Body(None)):
    """
    스냅샷은 /screenshots/{slug}/ 아래 저장됨.
    """
    user: Optional[str] = None
    if payload and isinstance(payload, dict):
        user = payload.get("user")

    ok = save_snapshot(user=user)
    return {"ok": ok, "user_slug": _user_slug}




# ----------------------------------------------------------
# WebSocket: 프레임 스트리밍
# ----------------------------------------------------------
@app.websocket("/ws/frame")
async def ws_frame(websocket: WebSocket):
    """
    WebSocket PUSH 버전:
      - 클라이언트 → 서버 : JPEG 바이너리 프레임
      - 서버 → 클라이언트 : live + session 상태 JSON
    기존 process_frame / get_live_state / get_session_meta 로직은 그대로 사용.
    """
    user = websocket.query_params.get("user") or None
    await websocket.accept()

    try:
        while True:
            # 클라이언트는 Blob(binary) 형태로 전송
            data = await websocket.receive_bytes()

            # JPEG → numpy → OpenCV 이미지 복원
            npimg = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # CPU-heavy process_frame 은 스레드풀에서 실행
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, process_frame, frame, user)

            # 처리 완료 후 최신 상태 스냅샷 추출
            live = get_live_state()
            session = get_session_meta()

            await websocket.send_json({
                "type": "update",
                "live": live,
                "session": session,
            })
    except WebSocketDisconnect:
        # 클라이언트 연결 종료
        return


# ----------------------------------------------------------
# Calibration Start
# ----------------------------------------------------------
@app.post("/api/calibrate/start")
def api_calibrate_start():
    ok = start_calibration()
    return {"ok": ok}

