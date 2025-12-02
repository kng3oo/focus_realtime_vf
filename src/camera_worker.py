import os
import time
import threading
from collections import deque
from typing import Dict, Any, Optional

import cv2
import numpy as np

from .model_utils import FocusPipeline, FocusResult
from .pose_utils import HeadPoseEstimator
from .calibration import Calibrator, Calib, save_calibration, load_calibration
from .metrics import EMA, RollingAverage
from .config import EMA_SEC_DISPLAY
from .log_utils import LogManager

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
SCREEN_DIR = os.path.join(PROJECT_ROOT, "screenshots")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(SCREEN_DIR, exist_ok=True)


# ----------------------------------------------------------
# 사용자 폴더 관리
# ----------------------------------------------------------

def slugify_korean(text: str) -> str:
    import re
    mapping = {
        "가":"ga","나":"na","다":"da","라":"ra","마":"ma","바":"ba","사":"sa",
        "아":"a","자":"ja","차":"cha","카":"ka","타":"ta","파":"pa","하":"ha",
        "강":"kang","김":"kim","박":"park","정":"jung","이":"lee","최":"choi",
        "홍":"hong","길":"gil","동":"dong",
    }
    out = ""
    for ch in text:
        if ch in mapping:
            out += mapping[ch]
        elif ch.isalnum():
            out += ch.lower()
    return re.sub(r"[^a-z0-9]+", "", out.lower()) or "user"


_user_name = "anonymous"
_user_slug = "anonymous"
_user_run_dir = os.path.join(RUNS_DIR, _user_slug)
_user_screen_dir = os.path.join(SCREEN_DIR, _user_slug)

def _apply_user_dirs():
    os.makedirs(_user_run_dir, exist_ok=True)
    os.makedirs(_user_screen_dir, exist_ok=True)


def set_user_name(name: Optional[str]):
    global _user_name, _user_slug, _user_run_dir, _user_screen_dir

    if not name:
        _user_name = "anonymous"
        _user_slug = "anonymous"
    else:
        _user_name = name
        _user_slug = slugify_korean(name)

    _user_run_dir = os.path.join(RUNS_DIR, _user_slug)
    _user_screen_dir = os.path.join(SCREEN_DIR, _user_slug)
    _apply_user_dirs()


# ----------------------------------------------------------
# Shared State
# ----------------------------------------------------------

shared_state: Dict[str, Any] = dict(
    lock=threading.Lock(),
    time=deque(maxlen=5000),
    focus=deque(maxlen=5000),
    emotion=deque(maxlen=5000),
    blink=deque(maxlen=5000),
    gaze=deque(maxlen=5000),
    neck=deque(maxlen=5000),
    latest={},
    frames=0,
    saved=0,
    fps=0.0,
    start_ts=time.strftime("%Y-%m-%d %H:%M:%S"),
    avg_10s=0.0,
    avg_1m=0.0,
    avg_10m=0.0,
    calibrating=False,
    calib_target=None,
)

# ----------------------------------------------------------
# Pipeline & Metrics
# ----------------------------------------------------------

_pipeline = FocusPipeline()

ema_focus = EMA(half_life_sec=EMA_SEC_DISPLAY, fps_guess=10.0)
roll10 = RollingAverage(seconds=10, fps_guess=10.0)
roll60 = RollingAverage(seconds=60, fps_guess=10.0)
roll600 = RollingAverage(seconds=600, fps_guess=10.0)

_last_frame_bgr: Optional[np.ndarray] = None
_last_result: Optional[FocusResult] = None
_last_save_ts = 0.0
_frame_count = 0
_last_fps_ts = time.time()


# ----------------------------------------------------------
# Calibration
# ----------------------------------------------------------

_calibrator = Calibrator()


def _load_user_calibration():
    path = os.path.join(_user_run_dir, "calib.json")
    if os.path.exists(path):
        c = load_calibration(path)
        if c:
            _pipeline.set_calibration(c)
            print(f"✅ Loaded calibration for {_user_slug}")


def _save_user_calibration(c: Calib):
    path = os.path.join(_user_run_dir, "calib.json")
    save_calibration(c, path)
    print(f"💾 Saved calibration → {path}")


# ----------------------------------------------------------
# Overlay & Image Save (flip ONLY HERE)
# ----------------------------------------------------------

def _render_overlay(frame: np.ndarray, res: FocusResult):
    # ONLY HERE: Flip 적용
    frame = cv2.flip(frame, 1)

    overlay = cv2.resize(frame, (1280, 960))
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 40
    cv2.putText(overlay, f"User: {_user_slug}", (20, y), font, 1.0, (255,255,255), 2); y += 40
    cv2.putText(overlay, f"Focus: {res.focus*100:.1f}%", (20, y),
                font, 1.0, (0,255,255), 2)
    return overlay


def _save_image(frame_bgr: np.ndarray, res: FocusResult, force=False):
    global _last_save_ts

    now = time.time()
    if not force:
        if res.focus >= 0.4:
            return False
        if now - _last_save_ts < 5.0:
            return False

    ts = time.strftime("%Y%m%d_%H%M%S")
    pct = int(res.focus * 100)
    filename = f"{_user_slug}_focus{pct}_{ts}.jpg"

    path = os.path.join(_user_screen_dir, filename)

    # ONLY HERE: Flip 적용
    overlay = _render_overlay(frame_bgr, res)
    cv2.imwrite(path, overlay)

    with shared_state["lock"]:
        shared_state["saved"] += 1

    _last_save_ts = now
    print(f"[Saved] {path}")
    return True


# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------

log_manager: Optional[LogManager] = None


def _init_user_log():
    """로그 파일 1개만 생성하고 append 유지"""
    global log_manager
    log_manager = LogManager(_user_run_dir)


def _save_log(elapsed: float, r: FocusResult):
    if log_manager:
        log_manager.write_row(
            focus=r.focus,
            gaze=r.gaze,
            neck=r.neck,
            yaw=r.yaw, pitch=r.pitch, roll=r.roll,
            ear=(r.ear or 0.0),
            emotion=r.top_emotion or "",
            blink=r.blink_score,
        )


# ----------------------------------------------------------
# Frame Processing (NO FLIP HERE)
# ----------------------------------------------------------

_last_user_slug = None

def process_frame(frame_bgr: np.ndarray, user: Optional[str] = None):
    """⚠️ 미러링 금지: raw frame 그대로 처리."""
    global _last_frame_bgr, _last_result, _frame_count, _last_fps_ts, _last_user_slug

    # 최초 실행 or 사용자 변경 시에만 초기화
    if user:
        set_user_name(user)

        if _last_user_slug != _user_slug:
            print(f"🔄 사용자 변경 감지 → 로그/캘리브레이션 초기화: {_user_slug}")
            _init_user_log()              # 로그 파일 1개 생성 (세션용)
            _load_user_calibration()      # 사용자별 캘리브레이션 로드
            _last_user_slug = _user_slug

    t_abs = time.time()

    # Calibration 진행 중
    if _calibrator.is_running():
        ok, yaw, pitch, roll, _ = HeadPoseEstimator().infer(frame_bgr)
        if ok:
            c = _calibrator.update(yaw, pitch, roll)
            with shared_state["lock"]:
                shared_state["calibrating"] = True
                shared_state["calib_target"] = _calibrator.current_target()

            if c is not None:
                _pipeline.set_calibration(c)
                _save_user_calibration(c)
                with shared_state["lock"]:
                    shared_state["calibrating"] = False
                    shared_state["calib_target"] = None
                print("✅ Calibration Completed")

    # MAIN PIPELINE (no flip)
    res: FocusResult = _pipeline.process(frame_bgr)
    _last_frame_bgr = frame_bgr.copy()
    _last_result = res

    # EMA / RollingAvg
    f_disp = ema_focus.update(res.focus)
    avg10 = roll10.update(f_disp)
    avg60 = roll60.update(f_disp)
    avg600 = roll600.update(f_disp)

    elapsed = t_abs - _pipeline.start_ts

    _save_log(elapsed, res)
    _save_image(frame_bgr, res, force=False)

    # FPS 계산
    _frame_count += 1
    dt = t_abs - _last_fps_ts
    if dt >= 1.0:
        with shared_state["lock"]:
            shared_state["fps"] = _frame_count / dt
        _frame_count = 0
        _last_fps_ts = t_abs

    # Shared State 업데이트
    with shared_state["lock"]:
        shared_state["time"].append(elapsed)
        shared_state["focus"].append(f_disp)
        shared_state["emotion"].append(res.emotion_score)
        shared_state["blink"].append(res.blink_score)
        shared_state["gaze"].append(res.gaze)
        shared_state["neck"].append(res.neck)
        shared_state["frames"] += 1
        shared_state["avg_10s"] = avg10
        shared_state["avg_1m"] = avg60
        shared_state["avg_10m"] = avg600
        shared_state["latest"] = dict(
            user=_user_name,
            focus=res.focus,
            gaze=res.gaze,
            neck=res.neck,
            emotion_score=res.emotion_score,
            blink_score=res.blink_score,
            ear=res.ear,
            blink_rate=res.blink_rate,
            yaw=res.yaw, pitch=res.pitch, roll=res.roll,
            top_emotion=res.top_emotion,
            quality=res.quality,
        )


def save_snapshot(user=None):
    if user:
        set_user_name(user)
        _init_user_log()
    if _last_frame_bgr is None or _last_result is None:
        return False
    return _save_image(_last_frame_bgr, _last_result, force=True)


def get_live_state():
    with shared_state["lock"]:
        return dict(
            time=list(shared_state["time"]),
            focus=list(shared_state["focus"]),
            emotion=list(shared_state["emotion"]),
            blink=list(shared_state["blink"]),
            gaze=list(shared_state["gaze"]),
            neck=list(shared_state["neck"]),
            latest=shared_state["latest"],
            fps=shared_state["fps"],
            avg_10s=shared_state["avg_10s"],
            avg_1m=shared_state["avg_1m"],
            avg_10m=shared_state["avg_10m"],
            calibrating=shared_state["calibrating"],
            calib_target=shared_state["calib_target"],
        )


def get_session_meta():
    with shared_state["lock"]:
        return dict(
            start_ts=shared_state["start_ts"],
            frames=shared_state["frames"],
            saved=shared_state["saved"],
        )


def start_calibration():
    _calibrator.start()
    with shared_state["lock"]:
        shared_state["calibrating"] = True
        shared_state["calib_target"] = _calibrator.current_target()
    return True

