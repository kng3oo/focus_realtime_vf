# 📘 집중도우미 AI튜터  

웹캠 기반 실시간 집중도·시선·목자세·감정·눈깜빡임 분석 시스템

본 프로젝트는 브라우저에서 촬영된 프레임을 처리하여  
머리자세, 시선 방향, 감정 확률, 눈깜빡임 빈도 등을 계산하고  
이들을 가중 평균하여 “집중도(Focus)” 점수를 산출합니다.


---

# 📁 프로젝트 구조

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

---

# 🔧 핵심 구성 요소 설명

## 1. FocusPipeline (model_utils.py)
집중도 계산의 핵심 로직.

- Head Pose 추정 (yaw, pitch, roll)
- EmotionModel (ConvNeXt Tiny 기반 감정 분류)
- BlinkEstimator(EAR 기반 눈깜빡임 검출)
- Gaze Score (yawㆍpitch 기준)
- Neck Score (yawㆍpitchㆍroll 기준)
- Quality Score로 전체 지표 보정
- 가중치 기반 F = 0.4G + 0.3N + 0.2E + 0.1B

## 2. HeadPoseEstimator (pose_utils.py)
MediaPipe FaceMesh + solvePnP 조합  
→ yaw, pitch, roll 추출  
→ 얼굴 크기 기반 quality 계산

## 3. EmotionModel (ConvNeXt Tiny)
- best_model.pth 로드  
- 7가지 감정 확률 계산  
- 부정 감정 비율로 emotion_score 산출

## 4. BlinkEstimator (EAR 기반)
- MediaPipe FaceMesh 사용  
- EAR < ear_close → 눈감음  
- EAR > ear_open → 눈뜸  
- blink_rate 계산

## 5. Calibration (calibration.py)
사용자가  
UL → UR → LR → LL → CTR  
총 5개 포인트를 1초씩 바라보며 기준각을 수집.  
결과는 calib.json으로 저장.

## 6. camera_worker.py
전체 파이프라인을 연결하는 모듈.

- 사용자별 폴더 생성  
- 캘리브레이션 자동 업데이트  
- FocusPipeline 호출  
- EMA / Rolling Avg 계산  
- 로그 CSV 기록  
- 자동 스냅샷 저장

## 7. index.html (대시보드)
- 실시간 Focus/Gaze/Neck/Emotion/Blink 표시  
- 감정 빈도수 랭킹  
- 사용자 이름 입력  
- 캘리브레이션 진행 상태 표시  
- WebSocket 기반 실시간 반영  
- 카메라 영상은 로컬 전용

## 8. log_utils.py
세션 중 CSV 파일 1개를 생성하고 append 방식으로 기록.

컬럼:
timestamp, focus, gaze, neck, yaw, pitch, roll, ear, emotion, blink

## 9. config.py
가중치 및 파이프라인 파라미터:
WEIGHTS = {gaze:0.4, neck:0.3, emotion:0.2, blink:0.1}
QUALITY_EXPONENT = 0.7
CALIB_HOLD_SEC = 2.0

---

# 🧠 집중도 계산 공식

F = 0.4 * Gaze  
  + 0.3 * Neck  
  + 0.2 * Emotion  
  + 0.1 * Blink

---

# 🖥 처리 흐름 요약

브라우저 → WebSocket → 서버 수신  
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
집중도 계산 → 클라이언트로 실시간 전송  
