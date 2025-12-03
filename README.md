# 📘 집중도우미 AI튜터  
웹캠 기반 **실시간 집중도·시선·목자세·감정·눈깜빡임 분석 시스템**

브라우저에서 촬영된 프레임을 서버로 전송하면  
AI 파이프라인이 이를 분석하여 집중도(Focus) 점수를 실시간 산출합니다.

---

# 🚀 주요 기능

- 실시간 Head Pose 추정 (yaw, pitch, roll)
- 시선(Gaze) 분석
- 목자세(Neck) 안정성 분석
- ConvNeXt Tiny 기반 감정 분석
- EAR 기반 눈깜빡임(Blink) 검출
- 사용자별 캘리브레이션(5-Point)
- EMA/Rolling Avg 기반 노이즈 보정
- CSV 기반 집중 로그 자동 저장
- 포커스 낮을 때 스냅샷 자동 저장
- WebSocket 실시간 대시보드

---

# 📁 프로젝트 구조

```
project/
│── static/
│    └── index.html        # 실시간 대시보드 UI
│
│── models/
│    └── best_model.pth    # 감정 분류 ConvNeXt Tiny 모델
│
│── runs/                  # 사용자별 로그 CSV 저장
│── screenshots/           # 포커스 낮을 때 자동 저장되는 이미지
│
│── main.py                # FastAPI WebSocket 서버
│── web_server.py          # 대시보드 서버 버전
│── camera_worker.py       # 실시간 영상 처리 파이프라인
│── model_utils.py         # Emotion/Gaze/Neck/Blink/Focus 계산
│── pose_utils.py          # MediaPipe 기반 머리 자세 추정
│── calibration.py         # 5-Point Calibration 시스템
│── metrics.py             # EMA / Rolling Average
│── config.py              # 가중치 및 파라미터 설정
│── log_utils.py           # CSV 로그 매니저
```

---

# 🔧 핵심 구성 요소

## 1️⃣ FocusPipeline (`model_utils.py`)
전체 집중도 산출 로직을 담당합니다.

- Head Pose (yaw, pitch, roll)
- EmotionModel (감정 확률 → 점수 변환)
- BlinkEstimator(EAR 기반 눈깜빡임)
- Gaze Score
- Neck Score
- Quality 보정
- 가중 합산:  
  **F = 0.4G + 0.3N + 0.2E + 0.1B**

---

## 2️⃣ HeadPoseEstimator (`pose_utils.py`)
📌 **MediaPipe FaceMesh + SolvePnP 기반**

- 얼굴 랜드마크 추출  
- 3D 모델 포인트 매칭  
- yaw/pitch/roll 계산  
- 얼굴 크기 → quality 산출  

---

## 3️⃣ EmotionModel (ConvNeXt Tiny)
- `best_model.pth` 로드
- 7가지 감정 확률 계산
- 부정 감정 비율로 emotion_score 산출

---

## 4️⃣ BlinkEstimator (EAR 기반)
- MediaPipe FaceMesh 랜드마크 기반 EAR 계산  
- EAR < ear_close → 눈 감김  
- EAR > ear_open → 눈 뜸  
- blink_rate 산출  

---

## 5️⃣ Calibration (`calibration.py`)
**사용자 5-Point 시선 기준 수집**

```
UL → UR → LR → LL → CTR
각 1초씩 고정 → calib.json 저장
```

---

## 6️⃣ camera_worker.py
전체 파이프라인을 연결하는 모듈:

- 사용자 폴더 자동 생성
- 캘리브레이션 상태 관리
- FocusPipeline 호출
- EMA / Rolling Avg 적용
- CSV 로그 기록
- 스냅샷 자동 저장

---

## 7️⃣ index.html (대시보드)
- 실시간 Focus/Gaze/Neck/Emotion/Blink 표시  
- 감정 빈도 랭킹  
- 사용자 이름 입력 및 세션 관리  
- 캘리브레이션 진행 표시  
- WebSocket 기반 실시간 반영  
- 영상은 로컬 전용  

---

## 8️⃣ log_utils.py
CSV append 기반 데이터 저장

**컬럼:**
```
timestamp, focus, gaze, neck, yaw, pitch, roll, ear, emotion, blink
```

---

## 9️⃣ config.py
```
WEIGHTS = {
  gaze: 0.4,
  neck: 0.3,
  emotion: 0.2,
  blink: 0.1
}

QUALITY_EXPONENT = 0.7
CALIB_HOLD_SEC = 2.0
```

---

# 🧠 집중도 계산 공식

```
F = 0.4 * Gaze  
  + 0.3 * Neck  
  + 0.2 * Emotion  
  + 0.1 * Blink
```

---

# 🖥 처리 흐름 요약

```
브라우저 → WebSocket → 서버  
        ↓  
camera_worker.process_frame()  
        ↓  
FocusPipeline.process()  
        ├─ Head Pose  
        ├─ Emotion  
        ├─ Blink  
        ├─ Gaze Score  
        ├─ Neck Score  
        ↓  
집중도 계산 → 클라이언트 실시간 전송
```

---

# 💡 사용 예시 (서버 실행)

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

웹 대시보드:
```
python web_server.py
```

---

# 📜 License
MIT License
