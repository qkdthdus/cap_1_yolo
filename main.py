from ultralytics import YOLO
import cv2
import numpy as np
import pygame

# --------------------------
# 설정
# --------------------------
TRIGGER_BOX_SIZE = 300       # 사람 바운딩 박스 기준
VIDEO_PATH = "./brand_pic/video2.mp4"  # 재생할 영상

# --------------------------
# YOLO 모델 (사람)
model = YOLO("yolov8n.pt")

# --------------------------
# Pygame 초기화
pygame.init()
display_windows = []
for size in pygame.display.get_desktop_sizes():
    w, h = size
    win = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
    display_windows.append((win, w, h))

def show_frame_all(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for win, w, h in display_windows:
        surf = pygame.surfarray.make_surface(np.flip(frame_rgb, axis=1).swapaxes(0,1))
        surf = pygame.transform.scale(surf, (w, h))
        win.blit(surf, (0,0))
        pygame.display.update()

# --------------------------
# 사람 감지
def detect_person(frame):
    person_close = False
    boxes = []
    try:
        results = model(frame, verbose=False)
    except:
        return False, boxes

    for r in results:
        for det in r.boxes:
            cls = int(det.cls[0])
            if cls != 0:  # 사람 클래스
                continue
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
            w = x2 - x1
            h = y2 - y1
            if w >= TRIGGER_BOX_SIZE or h >= TRIGGER_BOX_SIZE:
                person_close = True
    return person_close, boxes

# --------------------------
# 손 영역 추정 (얼굴 제외)
def detect_hand(frame, person_boxes):
    """
    사람 박스 상단 35%를 얼굴 영역으로 가정하여 제외하고
    중앙 근처 밝은 영역/움직임을 손으로 추정
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        cx, cy = x + cw//2, y + ch//2
        
        # 사람 박스 내부 확인
        for px1, py1, px2, py2 in person_boxes:
            # 얼굴 영역 제외: 상단 35%
            face_limit_y = py1 + int((py2 - py1) * 0.35)
            if px1 <= cx <= px2 and cy > face_limit_y:
                # 화면 중앙 근처 체크
                if abs(cx - center[0]) < w*0.3 and abs(cy - center[1]) < h*0.3:
                    return True
    return False

# --------------------------
# 웹캠
cap = cv2.VideoCapture(0)
docent_active = False

# 영상 준비
cap_vid = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap_vid.read()
if not ret:
    raise Exception("영상 불러오기 실패")
cap_vid.release()

# 초기 화면: 영상 첫 프레임
show_frame_all(first_frame)

# --------------------------
# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    person_close, person_boxes = detect_person(frame)
    hand_detected = detect_hand(frame, person_boxes)

    # --- 트리거: 사람 + 손 영역
    if not docent_active and person_close and hand_detected:
        print("트리거 활성화: 영상 재생 시작")
        docent_active = True

        cap_vid = cv2.VideoCapture(VIDEO_PATH)
        video_fps = cap_vid.get(cv2.CAP_PROP_FPS)
        clock = pygame.time.Clock()

        while cap_vid.isOpened():
            ret_vid, vid_frame = cap_vid.read()
            if not ret_vid:
                break

            show_frame_all(vid_frame)

            # FPS 맞춰서 재생
            clock.tick(video_fps)

            # Pygame 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap_vid.release()
                    docent_active = False
                    show_frame_all(first_frame)

        cap_vid.release()
        docent_active = False
        show_frame_all(first_frame)  # 영상 끝나면 첫 프레임으로
        print("영상 재생 종료, 대기 상태로 복귀")

    # --- 디버깅용 화면
    debug_frame = frame.copy()
    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0,255,0), 2)
    if hand_detected:
        cv2.putText(debug_frame, "Hand detected!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Debug View", cv2.resize(debug_frame, (debug_frame.shape[1]//2, debug_frame.shape[0]//2)))

    # ESC 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
