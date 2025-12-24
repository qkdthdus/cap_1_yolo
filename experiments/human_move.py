"""
Advanced Movement Pattern Analyzer using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import math
import argparse
import sys

# =============================================================================
# Configuration & Constants
# =============================================================================
class Config:
    # Model
    MODEL_PATH = 'yolov8n.pt'
    CONF_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.5
    
    # Analysis Parameters
    HISTORY_LEN = 60          # 프레임 단위 과거 기록 저장 길이
    SMOOTHING_FACTOR = 0.7    # 이동 평균 필터 계수 (좌표 떨림 보정)
    
    # Behavior Thresholds (Pixel units, assume 1920x1080 or similar)
    SPEED_RUNNING = 15.0      # 프레임당 픽셀 이동량 (Running 기준)
    SPEED_WALKING = 2.0       # 프레임당 픽셀 이동량 (Walking 기준)
    
    # Loitering (배회) Logic
    LOITERING_RADIUS = 100    # 배회로 간주할 반경 (픽셀)
    LOITERING_TIME = 3.0      # 배회 판단 기준 시간 (초)
    
    # Visualization
    TRACE_COLOR = (0, 255, 255)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

# =============================================================================
# Helper Math Classes
# =============================================================================
class VectorMath:
    @staticmethod
    def euclidean_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def calculate_velocity(history):
        """
        최근 N 프레임 간의 평균 속도를 계산합니다.
        """
        if len(history) < 2:
            return 0.0
        # 최근 5프레임의 평균 이동량 계산
        recent = list(history)[-5:]
        dists = [VectorMath.euclidean_dist(recent[i], recent[i+1]) for i in range(len(recent)-1)]
        return sum(dists) / len(dists) if dists else 0.0

    @staticmethod
    def calculate_variance(history):
        """
        좌표들의 분산을 계산하여 한 곳에 머무르는지 확인합니다.
        """
        if len(history) < 10:
            return 9999 # Not enough data
        pts = np.array(history)
        std_dev = np.std(pts, axis=0) # [std_x, std_y]
        return np.mean(std_dev) # 평균 표준편차

# =============================================================================
# Subject Class (Individual Person Tracker)
# =============================================================================
class TrackedSubject:
    def __init__(self, track_id, initial_pos):
        self.track_id = track_id
        
        # Position Handling
        self.raw_pos = initial_pos
        self.smooth_pos = initial_pos
        self.history = deque(maxlen=Config.HISTORY_LEN)
        self.history.append(initial_pos)
        
        # State Variables
        self.state = "Initializing" # Initializing, Idle, Walking, Running, Loitering
        self.start_time = time.time()
        self.last_seen = time.time()
        
        # Loitering Specific
        self.loiter_start_time = None
        self.center_of_activity = initial_pos
    
    def update(self, new_pos):
        """
        매 프레임마다 객체의 위치를 업데이트하고 상태를 재평가합니다.
        """
        self.last_seen = time.time()
        self.raw_pos = new_pos
        
        # 1. Coordinate Smoothing (Exponential Moving Average)
        sx, sy = self.smooth_pos
        nx, ny = new_pos
        alpha = Config.SMOOTHING_FACTOR
        
        smoothed_x = alpha * nx + (1 - alpha) * sx
        smoothed_y = alpha * ny + (1 - alpha) * sy
        self.smooth_pos = (int(smoothed_x), int(smoothed_y))
        
        self.history.append(self.smooth_pos)
        
        # 2. Update Behavior State
        self._analyze_behavior()

    def _analyze_behavior(self):
        """
        속도와 위치 분산을 기반으로 행동을 결정합니다.
        """
        velocity = VectorMath.calculate_velocity(self.history)
        spread = VectorMath.calculate_variance(self.history)
        
        # A. Movement Check
        if velocity > Config.SPEED_RUNNING:
            self.state = "RUNNING (Abnormal)"
            self.loiter_start_time = None # Reset loitering
        elif velocity > Config.SPEED_WALKING:
            self.state = "Walking"
            self.loiter_start_time = None
        else:
            # B. Static / Loitering Check
            # 움직임이 적고(속도 낮음), 활동 반경(spread)이 좁은 경우
            if spread < Config.LOITERING_RADIUS / 2:
                if self.loiter_start_time is None:
                    self.loiter_start_time = time.time()
                
                elapsed = time.time() - self.loiter_start_time
                if elapsed > Config.LOITERING_TIME:
                    self.state = f"LOITERING ({int(elapsed)}s)"
                else:
                    self.state = "Idle"
            else:
                self.state = "Micro-Movement"

    def draw(self, frame):
        """
        객체의 상태 정보를 프레임에 시각화합니다.
        """
        cx, cy = self.smooth_pos
        
        # Color Coding based on State
        if "RUNNING" in self.state:
            color = (0, 0, 255) # Red
        elif "LOITERING" in self.state:
            color = (0, 165, 255) # Orange
        elif "Walking" in self.state:
            color = (0, 255, 0) # Green
        else:
            color = (200, 200, 200) # Gray
            
        # Draw Trajectory
        pts = np.array(list(self.history), np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, color, 2)
        
        # Draw Label
        label = f"ID:{self.track_id} | {self.state}"
        cv2.putText(frame, label, (cx - 40, cy - 20), Config.TEXT_FONT, 0.6, color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

# =============================================================================
# Main Processor Class
# =============================================================================
class MovementAnalyzer:
    def __init__(self, source):
        print(f"[Init] Loading YOLOv8 model from {Config.MODEL_PATH}...")
        self.model = YOLO(Config.MODEL_PATH)
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"[Error] Could not open video source: {source}")
            sys.exit(1)
            
        self.subjects = {} # Dictionary to store TrackedSubject objects
        
    def run(self):
        print("[Run] Starting analysis loop...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 1. Object Tracking (using BoT-SORT or ByteTrack provided by Ultralytics)
            # persist=True maintains IDs between frames
            results = self.model.track(frame, persist=True, verbose=False, classes=[0], tracker="bytetrack.yaml")
            
            current_frame_ids = []
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    current_frame_ids.append(track_id)
                    
                    # Register or Update Subject
                    if track_id not in self.subjects:
                        self.subjects[track_id] = TrackedSubject(track_id, (cx, cy))
                    else:
                        self.subjects[track_id].update((cx, cy))
                    
                    # Draw Bounding Box
                    color = (0, 255, 0)
                    if "RUNNING" in self.subjects[track_id].state:
                         color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 2. Cleanup Old Tracks
            # 현재 프레임에 감지되지 않은 객체 중 오래된 것은 삭제
            active_subjects = {}
            for tid, subj in self.subjects.items():
                if time.time() - subj.last_seen < 2.0: # 2초 동안 안보이면 삭제
                    active_subjects[tid] = subj
                    if tid in current_frame_ids:
                        subj.draw(frame)
            self.subjects = active_subjects

            # 3. Global Info
            cv2.putText(frame, f"Active Targets: {len(self.subjects)}", (20, 40), 
                        Config.TEXT_FONT, 1, (0, 255, 255), 2)
            
            cv2.imshow("Advanced Behavior Analysis", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # If video file exists, use it. Otherwise use webcam.
    import os
    target_source = "video1.mp4" if os.path.exists("video1.mp4") else 0
    
    analyzer = MovementAnalyzer(target_source)
    analyzer.run()