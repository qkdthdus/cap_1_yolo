from core.camera import Camera
from core.inference import InferenceEngine
from core.gesture import HandFSM
from core.trigger import TriggerManager
from core.screen import ScreenUnit
import cv2

VIDEOS = [
    "./brand_pic/video1.mp4",
    "./brand_pic/video2.mp4",
    "./brand_pic/video3.mp4",
    "./brand_pic/video4.mp4",
]

cam = Camera()
infer = InferenceEngine()
fsm = HandFSM()
trigger_mgr = TriggerManager(cooldown=3.0)

# 🔔 마우스 수동 트리거
mouse_trigger = False

def handle_mouse(event, x, y, flags, param):
    global mouse_trigger
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_trigger = True
        print("🖱️ Monitor 1 좌클릭 → 수동 트리거")

screens = [ScreenUnit(f"Monitor {i+1}", v) for i,v in enumerate(VIDEOS)]
cv2.setMouseCallback("Monitor 1", handle_mouse)

DEBUG_NAME = "DEBUG"
cv2.namedWindow(DEBUG_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(DEBUG_NAME, 640, 360)

frame_count = 0

# 🔒 상태 유지 변수 (중요)
person_close = False
hand_closed = False
hand_open = False


while True:
    frame = cam.read()
    if frame is None:
        break

    frame_count += 1

    # 🔍 YOLO는 가끔만
    if frame_count % 80 == 0:
        person_close, _ = infer.detect_person(frame)
        hand_closed, hand_open = infer.detect_hand(frame)

    # 🧠 FSM은 매 프레임
    ready = fsm.update(hand_closed, hand_open)

    if (person_close and ready or mouse_trigger) and trigger_mgr.can_trigger():
        for s in screens:
            s.play()
        trigger_mgr.fire()
        mouse_trigger = False  # 🔒 수동 트리거 소모


    for s in screens:
        s.update()

    # 🐞 DEBUG
    dbg = frame.copy()
    cv2.putText(dbg, f"CLOSE:{person_close}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(dbg, f"CLOSED:{hand_closed}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(dbg, f"OPEN:{hand_open}", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow(DEBUG_NAME, cv2.resize(dbg, (640,360)))

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
