from ultralytics import YOLO
import cv2
import numpy as np
import time
import datetime
# 모니터 해상도 정보를 가져오기 위해 tkinter 모듈 사용
import tkinter as tk

# --------------------------
# 전역 마우스 상태 변수
# --------------------------
mouse_r_click_triggered = False

# --------------------------
# 마우스 콜백 함수
# --------------------------
def handle_mouse_event(event, x, y, flags, param):
    """ 마우스 이벤트를 처리하고 우클릭 시 플래그를 설정합니다. """
    global mouse_r_click_triggered
    # 📢 마우스 오른쪽 버튼을 눌렀을 때 (cv2.EVENT_RBUTTONDOWN)
    if event == cv2.EVENT_RBUTTONDOWN:
        mouse_r_click_triggered = True
        print("=== 🖱️ 마우스 우클릭 수동 트리거 활성화! ===")


# --------------------------
# 모니터 해상도 가져오기 (Tkinter 사용)
# --------------------------
try:
    # Tkinter의 Toplevel 윈도우를 사용하여 화면 해상도를 가져옵니다.
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    print(f"시스템 해상도 감지: {screen_width}x{screen_height}")
except tk.TclError:
    print("경고: Tkinter 초기화 실패. 기본 해상도 1920x1080 사용.")
    screen_width = 1920
    screen_height = 1080

# --------------------------
# 설정 값 (모니터 해상도 기반으로 계산)
# --------------------------
VIDEO_FILES = [
    "./brand_pic/video1.mp4", 
    "./brand_pic/video2.mp4", 
    "./brand_pic/video3.mp4", 
    "./brand_pic/video4.mp4" 
]
TRIGGER_BOX_SIZE = 300
DEBUG_WINDOW_NAME = "Webcam Debug View (ESC to Quit)"

# 📢 추가: 쿨다운 설정
COOLDOWN_DURATION = 3.0 # 3초 쿨다운

# 초기 모니터 창 크기를 화면 해상도 기반으로 계산
MAX_ROW_SIZE = 2 # 한 줄에 2개 창 배치

# 배치 간격 및 시작 위치
MARGIN_X, MARGIN_Y = 20, 40 
START_X, START_Y = 50, 50

# 4개 창이 2x2로 배치될 수 있는 최대 크기 계산
INITIAL_WINDOW_W = (screen_width - START_X - (MAX_ROW_SIZE + 1) * MARGIN_X) // MAX_ROW_SIZE
INITIAL_WINDOW_H = (screen_height - START_Y - (MAX_ROW_SIZE + 1) * MARGIN_Y) // MAX_ROW_SIZE

# 최소 크기 제한 (너무 작아지는 것을 방지)
INITIAL_WINDOW_W = max(320, INITIAL_WINDOW_W)
INITIAL_WINDOW_H = max(180, INITIAL_WINDOW_H)

print(f"계산된 초기 창 크기: {INITIAL_WINDOW_W}x{INITIAL_WINDOW_H}")

# --------------------------
# YOLO 모델 로드 (생략)
# --------------------------
model = YOLO("yolov8n.pt") 

try:
    pose_model = YOLO("yolov8n-pose.pt")
except Exception as e:
    print("--- ⚠️ 경고: yolov8n-pose.pt 모델 로드 실패 ⚠️ ---")
    print(f"오류: {e}")
    print("YOLO Pose 모델을 다운로드하여 스크립트와 같은 경로에 두십시오.")
    pose_model = None 

# --------------------------
# 헬퍼 함수 (이전과 동일)
# --------------------------

def detect_person(frame):
    close = False
    boxes = []
    results = model(frame, classes=0, verbose=False) 
    for r in results:
        for det in r.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
            w = x2 - x1
            h = y2 - y1
            if w >= TRIGGER_BOX_SIZE or h >= TRIGGER_BOX_SIZE: 
                close = True
    return close, boxes

# 📢 주의: 이 함수는 프레임에 직접 시각화를 수행합니다.
def get_hand_status_pose(frame):
    if pose_model is None:
        return False, False

    # Pose 감지 시, 프레임이 이미 좌우 반전된 상태인지 고려해야 합니다.
    # YOLO 추론은 원본 프레임(반전되지 않은)으로 실행하는 것이 정확도를 높일 수 있지만,
    # 현재 구조상 반전된 프레임이 전달될 것이므로, Pose는 반전된 이미지 좌표에 맞춰 시각화합니다.
    pose_results = pose_model(frame, verbose=False)
    
    WRIST_KPTS = [9, 10]
    ELBOW_KPTS = [7, 8]
    CONF_THRESHOLD = 0.5 
    MIN_DISTANCE = 50 

    hand_is_open = False
    hand_is_closed = False

    for r in pose_results:
        if r.keypoints is None or r.keypoints.data.numel() == 0:
            continue
            
        kpts = r.keypoints.data[0].cpu().numpy() 
        if kpts.shape[0] < 17: continue
        
        h, w = frame.shape[:2]
        if r.boxes and r.boxes.xyxy.numel() > 0:
             x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())
             person_center_x = (x1 + x2) // 2
             cam_center_x = w // 2
             if abs(person_center_x - cam_center_x) > w * 0.4: continue


        for wrist_idx, elbow_idx in zip(WRIST_KPTS, ELBOW_KPTS):
            wrist_kpt = kpts[wrist_idx]
            elbow_kpt = kpts[elbow_idx]
            
            if wrist_kpt[2] > CONF_THRESHOLD and elbow_kpt[2] > CONF_THRESHOLD:
                
                wrist_pos = np.array([wrist_kpt[0], wrist_kpt[1]])
                elbow_pos = np.array([elbow_kpt[0], elbow_kpt[1]])
                distance = np.linalg.norm(wrist_pos - elbow_pos)
                
                if distance > MIN_DISTANCE:
                    hand_is_open = True
                    cv2.circle(frame, (int(wrist_kpt[0]), int(wrist_kpt[1])), 8, (0, 255, 0), -1) 
                else:
                    hand_is_closed = True
                    cv2.circle(frame, (int(wrist_kpt[0]), int(wrist_kpt[1])), 8, (0, 0, 255), -1)

    return hand_is_closed, hand_is_open


# --------------------------
# 1️⃣ OpenCV 창 생성 및 초기화 
# --------------------------
screen_units = []

current_x = START_X 
current_y = START_Y
max_h_in_row = 0

for i in range(len(VIDEO_FILES)):
    window_name = f"Video Monitor {i+1}"
    
    cap_vid = cv2.VideoCapture(VIDEO_FILES[i])
    if not cap_vid.isOpened():
         raise Exception(f"영상 불러오기 실패: {VIDEO_FILES[i]}")

    W_orig = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_orig = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = cap_vid.read()
    cap_vid.release()

    if not ret:
        raise Exception(f"영상 불러오기 실패: {VIDEO_FILES[i]}")

    # 1. 창 배치 위치 계산 
    if i % MAX_ROW_SIZE == 0 and i > 0: 
        current_y += INITIAL_WINDOW_H + MARGIN_Y
        current_x = START_X
        max_h_in_row = 0

    # 2. OpenCV 창 생성 및 위치/크기 지정 
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    
    # 계산된 INITIAL_WINDOW_W/H로 크기 설정
    cv2.resizeWindow(window_name, INITIAL_WINDOW_W, INITIAL_WINDOW_H) 
    cv2.moveWindow(window_name, current_x, current_y) 
    
    # 📢 Video Monitor 1에 마우스 콜백 설정
    if i == 0:
        cv2.setMouseCallback(window_name, handle_mouse_event) 


    # 다음 창 위치 업데이트
    current_x += INITIAL_WINDOW_W + MARGIN_X
    max_h_in_row = max(max_h_in_row, INITIAL_WINDOW_H) 


    screen_units.append({
        "win_name": window_name,
        "active": False,
        "first_frame": first_frame,
        "video_path": VIDEO_FILES[i],
        "video_cap": None,
        "fps": 30,
        "delay_ms": 1, 
        "index": i,
        "width_orig": W_orig,     
        "height_orig": H_orig,    
        "initial_w": INITIAL_WINDOW_W, 
        "initial_h": INITIAL_WINDOW_H,
    })

# 초기 화면 정지 상태 표시
for unit in screen_units:
    # 정지 화면 프레임을 계산된 초기 설정 크기로 리사이징하여 표시
    resized_frame = cv2.resize(unit["first_frame"], (unit["initial_w"], unit["initial_h"]))
    cv2.imshow(unit["win_name"], resized_frame)

# 디버그 창 생성 및 배치 (NORMAL 유지)
cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)

dbg_w, dbg_h = 640, 360
dbg_pos_x = current_x 
dbg_pos_y = START_Y 

cv2.resizeWindow(DEBUG_WINDOW_NAME, dbg_w, dbg_h)
cv2.moveWindow(DEBUG_WINDOW_NAME, dbg_pos_x, dbg_pos_y)


# --------------------------
# 2️⃣ 웹캠 및 메인 루프 
# --------------------------
cap = cv2.VideoCapture(0)
running = True

cooldown_end_time = 0.0

frame_counter = 0
INFERENCE_FREQUENCY = 20 

person_close = False
boxes = []
hand_is_closed = False
hand_is_open = False
hand_was_closed = False 

while running:
    frame_counter += 1
    
    ret, frame_raw = cap.read() # 📢 원본 프레임 읽기
    if not ret: break
    
    # 📢 좌우 반전 적용 (1: 좌우 반전)
    frame = cv2.flip(frame_raw, 1) 
    
    
    # 📢 조건부 YOLO 추론
    if frame_counter % INFERENCE_FREQUENCY == 0:
        # 참고: detect_person과 get_hand_status_pose는 반전된 'frame'을 사용합니다.
        person_close_current, boxes_current = detect_person(frame) 
        hand_is_closed_current, hand_is_open_current = get_hand_status_pose(frame) 
        
        person_close = person_close_current
        boxes = boxes_current
        hand_is_closed = hand_is_closed_current
        hand_is_open = hand_is_open_current
    
    
    # 쿨다운 상태 확인
    can_trigger = time.time() > cooldown_end_time
    
    # 최종 트리거 로직
    detection_trigger = person_close and hand_is_open and hand_was_closed and can_trigger
    mouse_trigger = mouse_r_click_triggered and can_trigger
    
    trigger = detection_trigger or mouse_trigger
    
    hand_was_closed = hand_is_closed
    
    
    # 5. 모니터 개별 처리 (트리거 로직 적용)
    
    # 마우스 트리거 초기화
    if mouse_r_click_triggered:
        mouse_r_click_triggered = False
        
    for unit in screen_units:
        
        # A) 트리거 발생 → 비디오 시작
        if trigger and not unit["active"]:
            unit["active"] = True
            unit["video_cap"] = cv2.VideoCapture(unit["video_path"])
            fps = unit["video_cap"].get(cv2.CAP_PROP_FPS)
            unit["fps"] = fps if fps > 0 else 30 
            
            unit["delay_ms"] = max(1, int(1000 / unit["fps"])) 


        # B) 재생 중이면 프레임 읽기
        if unit["active"]:
            ret_vid, vid_frame = unit["video_cap"].read()

            if ret_vid:
                w_current = cv2.getWindowImageRect(unit["win_name"])[2]
                h_current = cv2.getWindowImageRect(unit["win_name"])[3]

                if w_current > 0 and h_current > 0:
                    vid_frame = cv2.resize(vid_frame, (w_current, h_current))
                    
                cv2.imshow(unit["win_name"], vid_frame) 
                
            else:
                # 영상 끝 → 정지 화면 복귀
                unit["active"] = False
                if unit["video_cap"]:
                    unit["video_cap"].release()
                
                # 📢 쿨다운 설정: 모든 비디오가 끝나야만 쿨다운 시작
                if not any(u["active"] for u in screen_units):
                     cooldown_end_time = time.time() + COOLDOWN_DURATION
                     print(f"=== 🥶 쿨다운 시작: {COOLDOWN_DURATION}초 동안 인식 중지 ===")
                
                w_current = cv2.getWindowImageRect(unit["win_name"])[2]
                h_current = cv2.getWindowImageRect(unit["win_name"])[3]

                if w_current > 0 and h_current > 0:
                     resized_frame = cv2.resize(unit["first_frame"], (w_current, h_current))
                else:
                     resized_frame = cv2.resize(unit["first_frame"], (unit["initial_w"], unit["initial_h"]))

                cv2.imshow(unit["win_name"], resized_frame)

        # C) 재생 중 아니면 첫 화면 유지 
        elif not unit["active"]:
             w_current = cv2.getWindowImageRect(unit["win_name"])[2]
             h_current = cv2.getWindowImageRect(unit["win_name"])[3]
             
             if w_current != unit["initial_w"] or h_current != unit["initial_h"]:
                 if w_current > 0 and h_current > 0:
                    resized_frame = cv2.resize(unit["first_frame"], (w_current, h_current))
                    cv2.imshow(unit["win_name"], resized_frame)
    
    # 6. 디버깅 화면 및 키 입력 처리 
    dbg = frame.copy() # 📢 반전된 프레임을 복사하여 사용
    h_cam, w_cam = dbg.shape[:2]
    text_y_start = h_cam - 130
    
    # 쿨다운 남은 시간 계산
    remaining_cooldown = max(0.0, cooldown_end_time - time.time())

    for x1, y1, x2, y2 in boxes: 
        box_color = (0, 255, 255) if (x2 - x1) >= TRIGGER_BOX_SIZE or (y2 - y1) >= TRIGGER_BOX_SIZE else (255, 0, 0)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), box_color, 2)

    person_color = (0, 255, 255) if person_close else (100, 100, 100)
    open_color = (0, 255, 0) if hand_is_open and can_trigger else (0, 100, 100)
    was_closed_color = (0, 165, 255) if hand_was_closed else (100, 100, 100)

    cv2.putText(dbg, f"1. CLOSE: {person_close}", (10, text_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, person_color, 2)
    cv2.putText(dbg, f"2. (WAS) CLOSED: {hand_was_closed}", (10, text_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, was_closed_color, 2)
    cv2.putText(dbg, f"3. (IS) OPEN: {hand_is_open}", (10, text_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, open_color, 2)
    cv2.putText(dbg, f"INFERENCE: 1/{INFERENCE_FREQUENCY} Frames (MAX Speed)", (10, text_y_start + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    if trigger:
        trigger_color = (0, 0, 255)
        trigger_text = "🟢 TRIGGER ACTIVATED! (Dual Trigger)" 
    elif remaining_cooldown > 0:
        trigger_color = (128, 128, 128) # 회색
        trigger_text = f"⏳ COOLDOWN: {remaining_cooldown:.1f}s"
    elif mouse_r_click_triggered:
        trigger_color = (255, 165, 0) # 주황색 (마우스 대기)
        trigger_text = "WAITING FOR MOUSE TRIGGER"
    else:
        trigger_color = (255, 255, 255)
        trigger_text = "🔴 TRIGGER STANDBY"

    cv2.putText(dbg, trigger_text, (10, h_cam - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, trigger_color, 2)

    cv2.imshow(DEBUG_WINDOW_NAME, cv2.resize(dbg, (dbg_w, dbg_h)))
    
    
    key = cv2.waitKey(1) 
    
    if key & 0xFF == 27: # ESC 종료
        running = False
        
    if cv2.getWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        running = False


# --------------------------
# 종료
# --------------------------
cap.release()
cv2.destroyAllWindows()