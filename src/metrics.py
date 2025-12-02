# metrics.py
import time
import math
from collections import deque

def clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

class EMA:
    def __init__(self, half_life_sec: float, fps_guess: float = 10.0):
        # convert half-life (sec) to per-step alpha based on fps
        self.y = None
        self.set_half_life(half_life_sec, fps_guess)

    def set_half_life(self, half_life_sec: float, fps_guess: float):
        if half_life_sec <= 0:
            self.alpha = 1.0
        else:
            steps = max(1, int(fps_guess * half_life_sec))
            self.alpha = 1 - math.exp(math.log(0.5) / steps)

    def update(self, x: float) -> float:
        if self.y is None:
            self.y = x
        else:
            self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y

class RollingAverage:
    def __init__(self, seconds: float, fps_guess: float = 10.0):
        self.maxlen = max(1, int(seconds * fps_guess))
        self.buf = deque(maxlen=self.maxlen)

    def update(self, x: float) -> float:
        self.buf.append(x)
        return sum(self.buf) / len(self.buf)

