from ultralytics import YOLO
import numpy as np

TRIGGER_BOX_SIZE = 300

class InferenceEngine:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.pose_model = YOLO("yolov8n-pose.pt")

    def detect_person(self, frame):
        close = False
        boxes = []

        results = self.model(frame, classes=0, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                if (x2 - x1) >= TRIGGER_BOX_SIZE or (y2 - y1) >= TRIGGER_BOX_SIZE:
                    close = True

        return close, boxes

    def detect_hand(self, frame):
        hand_open = False
        hand_closed = False

        results = self.pose_model(frame, verbose=False)
        for r in results:
            if r.keypoints is None:
                continue

            kpts = r.keypoints.data[0].cpu().numpy()
            for wrist, elbow in [(9,7),(10,8)]:
                if kpts[wrist][2] > 0.5 and kpts[elbow][2] > 0.5:
                    dist = np.linalg.norm(kpts[wrist][:2] - kpts[elbow][:2])
                    if dist > 50:
                        hand_open = True
                    else:
                        hand_closed = True

        return hand_closed, hand_open
