from ultralytics import YOLO
import cv2
import numpy as np
import time # 비디오 재생 속도 조절을 위해 time 모듈 추가

# --------------------------
# 설정 값
# --------------------------
VIDEO_FILES = [
    "./brand_pic/video1.mp4", 
    "./brand_pic/video2.mp4", 
    "./brand_pic/video3.mp4", 
    "./brand_pic/video4.mp4" 
]
TRIGGER_BOX_SIZE = 300
DEBUG_WINDOW_NAME = "Webcam Debug View (ESC to Quit)"
# 창 배치 설정
WINDOW_W, WINDOW_H = 320, 180 
START_X, START_Y = 50, 50
FULLSCREEN_TOGGLE_KEY = ord('q') 

# --------------------------
# YOLO 모델 로드
# --------------------------
# YOLO 사람 모델 (바운딩 박스)
model = YOLO("yolov8n.pt") 

# YOLOv8 Pose 모델 로드 (손 포즈 추정을 위해)
try:
    # pose_model 로드 시도. 모델 파일이 없으면 오류 처리
    pose_model = YOLO("yolov8n-pose.pt")
except Exception as e:
    print("--- ⚠️ 경고: yolov8n-pose.pt 모델 로드 실패 ⚠️ ---")
    print(f"오류: {e}")
    print("YOLO Pose 모델을 다운로드하여 스크립트와 같은 경로에 두십시오.")
    pose_model = None 

# --------------------------
# 헬퍼 함수
# --------------------------

# 사람 감지 (YOLO) - 기존 유지
def detect_person(frame):
    close = False
    boxes = []
    # 0: person 클래스
    results = model(frame, classes=0, verbose=False) 
    for r in results:
        for det in r.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
            w = x2 - x1
            h = y2 - y1
            # 바운딩 박스 크기가 임계값 이상이면 가까운 것으로 간주
            if w >= TRIGGER_BOX_SIZE or h >= TRIGGER_BOX_SIZE: 
                close = True
    return close, boxes

# 🌟 수정된 손 상태 감지 함수 (주먹/펼침 추론)
def get_hand_status_pose(frame):
    """
    YOLOv8 Pose 모델을 사용하여 손목과 팔꿈치 거리를 기반으로 
    손이 펴진 상태(Open) 또는 주먹 상태(Closed)인지 추론합니다.
    """
    if pose_model is None:
        return False, False # (주먹 상태: False, 펴짐 상태: False)

    # Pose 감지 실행
    pose_results = pose_model(frame, verbose=False)
    
    # 키포인트 인덱스: 7: 팔꿈치(왼), 8: 팔꿈치(오), 9: 손목(왼), 10: 손목(오)
    WRIST_KPTS = [9, 10]
    ELBOW_KPTS = [7, 8]
    CONF_THRESHOLD = 0.5 
    # 손 상태 판단 임계값 (이 픽셀 거리보다 멀면 'Open'으로 간주)
    MIN_DISTANCE = 50 

    hand_is_open = False
    hand_is_closed = False

    for r in pose_results:
        if r.keypoints is None or r.keypoints.data.numel() == 0:
            continue
            
        kpts = r.keypoints.data[0].cpu().numpy() 
        if kpts.shape[0] < 17: continue
        
        # 1. 감지된 포즈가 프레임 중앙 근처에 있는 사람인지 확인
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())
        person_center_x = (x1 + x2) // 2
        cam_center_x = w // 2
        if abs(person_center_x - cam_center_x) > w * 0.4: continue

        # 2. 양쪽 팔/손 상태 분석
        for wrist_idx, elbow_idx in zip(WRIST_KPTS, ELBOW_KPTS):
            wrist_kpt = kpts[wrist_idx]
            elbow_kpt = kpts[elbow_idx]
            
            # 두 키포인트 모두 신뢰도 임계값 이상이어야 함
            if wrist_kpt[2] > CONF_THRESHOLD and elbow_kpt[2] > CONF_THRESHOLD:
                
                # 거리 계산
                wrist_pos = np.array([wrist_kpt[0], wrist_kpt[1]])
                elbow_pos = np.array([elbow_kpt[0], elbow_kpt[1]])
                distance = np.linalg.norm(wrist_pos - elbow_pos)
                
                # 손 펴짐 추론: 거리가 임계값 이상이면 'Open'으로 간주
                if distance > MIN_DISTANCE:
                    hand_is_open = True
                    # 디버깅용: 펴진 손에 녹색 원
                    cv2.circle(frame, (int(wrist_kpt[0]), int(wrist_kpt[1])), 8, (0, 255, 0), -1) 
                else:
                    # 거리가 임계값 이하면 'Closed'로 간주 (주먹 또는 웅크린 상태)
                    hand_is_closed = True
                    # 디버깅용: 주먹/웅크린 손에 빨간색 원
                    cv2.circle(frame, (int(wrist_kpt[0]), int(wrist_kpt[1])), 8, (0, 0, 255), -1)

    return hand_is_closed, hand_is_open

# --------------------------
# 1️⃣ OpenCV 창 생성 및 초기화 (창 관리 로직 유지)
# --------------------------
screen_units = []
is_fullscreen_mode = False

def toggle_fullscreen(unit_index):
    """ 지정된 창을 전체 화면 모드로 토글하고 상태를 업데이트합니다. """
    global screen_units, is_fullscreen_mode
    unit = screen_units[unit_index]

    is_fullscreen_mode = not is_fullscreen_mode 
    
    if is_fullscreen_mode:
        cv2.setWindowProperty(unit["win_name"], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(unit["win_name"], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        # 창 모드로 복귀 시 위치와 크기 재지정
        cv2.resizeWindow(unit["win_name"], WINDOW_W, WINDOW_H)
        
        row = unit_index // 2
        col = unit_index % 2
        pos_x = START_X + col * (WINDOW_W + 20)
        pos_y = START_Y + row * (WINDOW_H + 40)
        cv2.moveWindow(unit["win_name"], pos_x, pos_y)


for i in range(4): # 4개의 비디오 창 생성
    window_name = f"Video Monitor {i+1}"
    
    # 창 배치
    row = i // 2
    col = i % 2
    pos_x = START_X + col * (WINDOW_W + 20)
    pos_y = START_Y + row * (WINDOW_H + 40)
    
    # OpenCV 창 생성 및 위치 지정
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_W, WINDOW_H)
    cv2.moveWindow(window_name, pos_x, pos_y)

    # 비디오 파일의 첫 프레임 로드
    cap_vid = cv2.VideoCapture(VIDEO_FILES[i])
    ret, first_frame = cap_vid.read()
    cap_vid.release()

    if not ret:
        raise Exception(f"영상 불러오기 실패: {VIDEO_FILES[i]}")

    screen_units.append({
        "win_name": window_name,
        "active": False,
        "first_frame": first_frame,
        "video_path": VIDEO_FILES[i],
        "video_cap": None,
        "fps": 30,
        "index": i 
    })

# 초기 화면 정지 상태 표시
for unit in screen_units:
    resized_frame = cv2.resize(unit["first_frame"], (WINDOW_W, WINDOW_H))
    cv2.imshow(unit["win_name"], resized_frame)

# 디버그 창 생성 및 배치
cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(DEBUG_WINDOW_NAME, START_X + 2 * (WINDOW_W + 20), START_Y)


# --------------------------
# 2️⃣ 웹캠 및 메인 루프 (수정된 로직 적용)
# --------------------------
cap = cv2.VideoCapture(0)
running = True

# 🌟 손 상태 추적 변수: 이전 프레임에서 손이 주먹 상태였는지 추적
hand_was_closed = False 

while running:
    
    # 1. 카메라 입력
    ret, frame = cap.read()
    if not ret: break
    
    # 2. 사람 감지 (YOLO)
    person_close, boxes = detect_person(frame)

    # 3. 손 포즈 감지 및 상태 확인 (수정된 함수)
    # hand_is_closed: 현재 프레임에서 주먹 상태인가?
    # hand_is_open: 현재 프레임에서 펴짐 상태인가?
    hand_is_closed, hand_is_open = get_hand_status_pose(frame) 
    
    # 🌟 최종 트리거 조건 (주먹->필 때)
    # 1. 사람이 가까이 있고 (person_close)
    # 2. 현재 손이 펴진 상태이며 (hand_is_open)
    # 3. 직전 프레임에서는 주먹 상태였을 때 (hand_was_closed)
    trigger = person_close and hand_is_open and hand_was_closed

    # 4. 다음 프레임을 위한 상태 업데이트
    # 현재의 '주먹 상태'를 다음 루프의 '이전 주먹 상태'로 저장
    hand_was_closed = hand_is_closed
    
    # 5. 모니터 개별 처리 (트리거 로직 적용)
    for unit in screen_units:

        # A) 트리거 발생 → 비디오 시작
        if trigger and not unit["active"]:
            unit["active"] = True
            unit["video_cap"] = cv2.VideoCapture(unit["video_path"])
            unit["fps"] = unit["video_cap"].get(cv2.CAP_PROP_FPS) or 30 
            unit["delay_ms"] = int(1000 / unit["fps"])
            unit["start_time"] = time.time() # 비디오 재생 시간 기록

        # B) 재생 중이면 프레임 읽기
        if unit["active"]:
            # FPS에 맞춰 딜레이 계산
            elapsed_time = time.time() - unit["start_time"]
            # frame_delay = elapsed_time * unit["fps"]
            
            ret_vid, vid_frame = unit["video_cap"].read()

            if ret_vid:
                # 전체 화면이 아닐 때만 리사이징
                if not is_fullscreen_mode:
                    vid_frame = cv2.resize(vid_frame, (WINDOW_W, WINDOW_H))
                cv2.imshow(unit["win_name"], vid_frame)
            else:
                # 영상 끝 → 정지 화면 복귀
                unit["active"] = False
                if unit["video_cap"]:
                    unit["video_cap"].release()
                
                resized_frame = cv2.resize(unit["first_frame"], (WINDOW_W, WINDOW_H))
                cv2.imshow(unit["win_name"], resized_frame)

        # C) 재생 중 아니면 첫 화면 유지
        elif not unit["active"]:
            pass 

    
    # 6. 디버깅 화면 및 키 입력 처리
    dbg = frame.copy()
    h_cam, w_cam = dbg.shape[:2]
    
    # 디버깅 정보 표시
    if trigger:
        cv2.putText(dbg, "TRIGGER: 주먹 -> 펼침! (ON)", (10, h_cam - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    else:
        status_text = f"CLOSE: {person_close} / CLOSED: {hand_is_closed} / OPEN: {hand_is_open} / WAS_CLOSED: {hand_was_closed}"
        cv2.putText(dbg, f"TRIGGER: OFF ({status_text})", (10, h_cam - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # 디버그 창에 사람 감지 박스 그리기
    for x1, y1, x2, y2 in boxes: 
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
    cv2.imshow(DEBUG_WINDOW_NAME, cv2.resize(dbg, (dbg.shape[1]//2, dbg.shape[0]//2)))
    
    # 키 입력 감지
    key = cv2.waitKey(1)
    
    if key & 0xFF == 27: # ESC 종료
        running = False
    elif key == FULLSCREEN_TOGGLE_KEY: # 'q' 키 입력 시
        toggle_fullscreen(0) # 0번 모니터만 전체 화면 토글
    
    # 디버그 창이 닫히면 종료
    if cv2.getWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        running = False


# --------------------------
# 종료
# --------------------------
cap.release()
cv2.destroyAllWindows()
