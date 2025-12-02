# log_utils.py
import os
import csv
import time
from typing import Optional

class LogManager:
    """
    한 세션(프로그램 실행) 동안 하나의 CSV 파일만 생성하고 계속 append.
    """

    def __init__(self, base_dir: str):
        """
        base_dir: 사용자별 runs/{user_slug} 경로
        """
        os.makedirs(base_dir, exist_ok=True)

        # 세션 최초 1회 생성되는 파일명
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(base_dir, f"{ts}.csv")

        # 최초 생성 시 헤더 작성
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "focus", "gaze", "neck",
                    "yaw", "pitch", "roll",
                    "ear", "emotion", "blink"
                ])

        print(f"✅ Log file initialized: {self.csv_path}")

    def get_csv_path(self) -> str:
        """ 웹에서 다운로드할 수 있도록 반환 """
        return self.csv_path

    def write_row(
        self,
        focus: float,
        gaze: float,
        neck: float,
        yaw: float,
        pitch: float,
        roll: float,
        ear: float,
        emotion: str,
        blink: float
    ):
        """ 각 프레임 결과를 CSV에 한 줄씩 append """

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                focus, gaze, neck,
                yaw, pitch, roll,
                ear, emotion, blink
            ])

