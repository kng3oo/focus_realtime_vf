# calibration.py
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict

from .metrics import clip01
from .config import (
    CALIB_POINTS, CALIB_HOLD_SEC,
    ALLOW_MIN_YAW_DEG, ALLOW_MIN_PITCH_DEG, ALLOW_MIN_ROLL_DEG
)

@dataclass
class Calib:
    baseline_yaw: float = 0.0
    baseline_pitch: float = 0.0
    baseline_roll: float = 0.0
    yaw_allow: float = 10.0
    pitch_allow: float = 8.0
    roll_allow: float = 8.0

    def as_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: Dict) -> "Calib":
        return cls(**d)

def save_calibration(c: Calib, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(c.as_dict(), f, ensure_ascii=False, indent=2)

def load_calibration(path: str) -> Optional[Calib]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return Calib.from_dict(json.load(f))
    except Exception:
        return None

class Calibrator:
    """
    서버측 자동 5-포인트 캘리브레이션(UL→UR→LR→LL→CTR)
    외부에서 매 프레임 head pose(yaw,pitch,roll)를 넣어주면,
    내부 상태머신이 각 구간의 평균을 모으고 완료되면 Calib 반환
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.points = {p: {"sum": [0.0, 0.0, 0.0], "n": 0} for p in CALIB_POINTS}
        self.order = list(CALIB_POINTS)
        self.curr_idx = 0
        self.started_at = None
        self.done = False

    def start(self):
        self.reset()
        self.started_at = time.time()

    def is_running(self) -> bool:
        return (self.started_at is not None) and (not self.done)

    def current_target(self) -> Optional[str]:
        if self.done or self.started_at is None or self.curr_idx >= len(self.order):
            return None
        return self.order[self.curr_idx]

    def update(self, yaw: float, pitch: float, roll: float) -> Optional[Calib]:
        if not self.is_running():
            return None

        target = self.current_target()
        elapsed = time.time() - self.started_at

        # 현재 타겟 구간에 샘플 누적
        s = self.points[target]
        s["sum"][0] += yaw; s["sum"][1] += pitch; s["sum"][2] += roll
        s["n"] += 1

        # 구간 유지 시간이 지나면 다음 타겟으로
        if elapsed >= (self.curr_idx + 1) * CALIB_HOLD_SEC:
            self.curr_idx += 1
            if self.curr_idx >= len(self.order):
                # 완료 → 결과 산출
                ctr = self._avg(self.points["CTR"])
                ul  = self._avg(self.points["UL"])
                ur  = self._avg(self.points["UR"])
                ll  = self._avg(self.points["LL"])
                lr  = self._avg(self.points["LR"])

                baseline_yaw, baseline_pitch, baseline_roll = ctr

                yaw_allow = max(
                    abs(ur[0] - ctr[0]),
                    abs(ul[0] - ctr[0]),
                    abs(lr[0] - ctr[0]),
                    abs(ll[0] - ctr[0]),
                    ALLOW_MIN_YAW_DEG
                )
                pitch_allow = max(
                    abs(ul[1] - ctr[1]),
                    abs(ur[1] - ctr[1]),
                    abs(ll[1] - ctr[1]),
                    abs(lr[1] - ctr[1]),
                    ALLOW_MIN_PITCH_DEG
                )
                roll_allow = max(
                    abs(ul[2] - ctr[2]),
                    abs(ur[2] - ctr[2]),
                    abs(ll[2] - ctr[2]),
                    abs(lr[2] - ctr[2]),
                    ALLOW_MIN_ROLL_DEG
                )

                self.done = True
                return Calib(
                    baseline_yaw=baseline_yaw,
                    baseline_pitch=baseline_pitch,
                    baseline_roll=baseline_roll,
                    yaw_allow=yaw_allow,
                    pitch_allow=pitch_allow,
                    roll_allow=roll_allow,
                )
        return None

    @staticmethod
    def _avg(s):
        if s["n"] <= 0: return (0.0, 0.0, 0.0)
        return (s["sum"][0]/s["n"], s["sum"][1]/s["n"], s["sum"][2]/s["n"])

