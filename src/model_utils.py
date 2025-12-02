# model_utils.py
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms

try:
    import mediapipe as mp
except ImportError:
    mp = None

from .pose_utils import HeadPoseEstimator
from .calibration import Calib
from .metrics import clip01
from .config import (
    WEIGHTS, CAMERA_LOOK_UP_DEG, BOOK_LOOK_DOWN_DEG,
    CAMERA_PENALTY, BOOK_PENALTY, QUALITY_EXPONENT
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
META_CSV = os.path.join(PROJECT_ROOT, "meta.csv")

@dataclass
class FocusResult:
    focus: float
    gaze: float
    neck: float
    emotion_score: float
    blink_score: float
    ear: Optional[float]
    blink_rate: Optional[float]
    top_emotion: Optional[str]
    raw_probs: Dict[str, float]
    yaw: float
    pitch: float
    roll: float
    quality: float

# ---------------- Emotion ----------------
class EmotionModel:
    def __init__(self, device: Optional[str] = None, img_size: int = 384):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        if os.path.exists(META_CSV):
            df = pd.read_csv(META_CSV)
            labels = sorted(df["label"].unique())
        else:
            labels = ["anger","disgust","fear","happy","none","sad","surprise"]

        self.labels = labels
        self.num_classes = len(labels)

        self.model = timm.create_model("convnext_tiny", pretrained=False, num_classes=self.num_classes)
        if os.path.exists(MODEL_PATH):
            sd = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            if isinstance(sd, dict):
                if "model" in sd: sd = sd["model"]
                elif "state_dict" in sd: sd = sd["state_dict"]
            try:
                self.model.load_state_dict(sd, strict=False)
                print("✅ Emotion model loaded.")
            except Exception as e:
                print("⚠️ 모델 weights 로드 실패:", e)
        else:
            print("⚠️ best_model.pth 없음 → 감정 점수 dummy 반환됨.")

        self.model.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
        self.negative_labels = {"anger","disgust","fear","sad"}

    def _crop_center(self, bgr):
        h, w = bgr.shape[:2]
        side = min(h, w)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        return bgr[y1:y1+side, x1:x1+side]

    def score(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, float], Optional[str], float]:
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.5, {}, None, 0.0
        img = self._crop_center(frame_bgr)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
        probs = {lab: float(p) for lab, p in zip(self.labels, prob)}
        neg_sum = sum(probs.get(k, 0.0) for k in self.negative_labels)
        emotion_score = float(np.clip(1.0 - neg_sum, 0.0, 1.0))
        top = max(probs, key=probs.get) if probs else None
        # 간단 확신도: top 확률
        conf = probs[top] if top else 0.0
        return emotion_score, probs, top, float(conf)

# ---------------- Blink (EAR) ----------------
class BlinkEstimator:
    def __init__(self, ear_close=0.18, ear_open=0.30, hist_len=60):
        self.ear_close = ear_close
        self.ear_open = ear_open
        self.hist = []
        self.max_hist = hist_len
        self.blinks = 0
        self.prev_closed = False
        if mp:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
            self.left_idx = [33,160,158,133,153,144]
            self.right_idx = [263,387,385,362,380,373]
        else:
            self.face_mesh = None

    def _ear(self, e: np.ndarray) -> float:
        A = np.linalg.norm(e[1] - e[5])
        B = np.linalg.norm(e[2] - e[4])
        C = np.linalg.norm(e[0] - e[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def score(self, frame_bgr: np.ndarray) -> Tuple[float, Optional[float], Optional[float]]:
        if self.face_mesh is None:
            return 1.0, None, None
        h, w = frame_bgr.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return 0.5, None, None
        lm = res.multi_face_landmarks[0].landmark
        def xy(i): return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)
        L = np.array([xy(i) for i in self.left_idx])
        R = np.array([xy(i) for i in self.right_idx])
        ear = (self._ear(L) + self._ear(R)) / 2.0
        s = (ear - self.ear_close) / (self.ear_open - self.ear_close + 1e-6)
        frame_score = float(np.clip(s, 0.0, 1.0))
        closed = ear < self.ear_close
        self.hist.append(closed)
        if len(self.hist) > self.max_hist: self.hist.pop(0)
        if self.prev_closed and not closed: self.blinks += 1
        self.prev_closed = closed
        blink_rate = float(self.blinks * (60.0 / max(1, len(self.hist)))) if self.max_hist > 0 else None
        return frame_score, float(ear), blink_rate

# ---------------- Gaze/Neck from HeadPose + Calib ----------------

def gaze_score(yaw: float, pitch: float, c: Calib) -> float:
    # New simplified scoring: 0-1 based on closeness to baseline
    dy = abs(yaw - c.baseline_yaw)
    dp = abs(pitch - c.baseline_pitch)
    yaw_score = max(0.0, 1.0 - dy/30.0)
    pitch_score = max(0.0, 1.0 - dp/25.0)
    return clip01((yaw_score + pitch_score) / 2.0)



def neck_score(yaw: float, pitch: float, roll: float, c: Calib) -> float:
    dp = abs(pitch - c.baseline_pitch)
    dy = abs(yaw - c.baseline_yaw)
    dr = abs(roll - c.baseline_roll)
    p_score = max(0.0, 1.0 - dp/30.0)
    y_score = max(0.0, 1.0 - dy/30.0)
    r_score = max(0.0, 1.0 - dr/35.0)
    return clip01((p_score+y_score+r_score)/3.0)


# ---------------- Focus Pipeline ----------------
class FocusPipeline:
    def __init__(self):
        self.emotion = EmotionModel()
        self.blink = BlinkEstimator()
        self.pose = HeadPoseEstimator()
        self.start_ts = time.time()
        self.calib: Optional[Calib] = None

    def set_calibration(self, c: Calib):
        self.calib = c

    def process(self, frame_bgr: np.ndarray) -> FocusResult:
        # Head pose
        ok, yaw, pitch, roll, quality = self.pose.infer(frame_bgr)
        if not ok:
            # 품질 0이면 모든 점수 반쯤으로 보수 처리
            quality = 0.0

        # E
        emo_score, probs, top, conf = self.emotion.score(frame_bgr)
        # (선택) 확신도 보정
        emo_score = emo_score * (conf ** 0.6 if conf > 0 else 0.7)

        # B
        blink_score, ear, blink_rate = self.blink.score(frame_bgr)

        # G / N (캘리브 전이면 중립 0.5)
        if self.calib is None or quality <= 0.0:
            g = 0.5
            n = 0.5
        else:
            g = gaze_score(yaw, pitch, self.calib)
            n = neck_score(yaw, pitch, roll, self.calib)

        # 품질 보정
        qexp = QUALITY_EXPONENT
        for_x = lambda x: x * (quality ** qexp) if quality > 0 else x * 0.5
        g = for_x(g); n = for_x(n); e = for_x(emo_score); b = for_x(blink_score)

        # Final
        F = (
            WEIGHTS["gaze"]   * g +
            WEIGHTS["neck"]   * n +
            WEIGHTS["emotion"]* e +
            WEIGHTS["blink"]  * b
        )
        F = float(clip01(F))

        return FocusResult(
            focus=F, gaze=g, neck=n, emotion_score=e, blink_score=b,
            ear=ear, blink_rate=blink_rate, top_emotion=top, raw_probs=probs,
            yaw=yaw, pitch=pitch, roll=roll, quality=quality
        )

