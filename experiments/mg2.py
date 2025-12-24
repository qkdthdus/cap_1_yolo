"""
Background Subtraction & Blob Tracking
"""

import cv2
import numpy as np
import math

class Config:
    # MOG2 Parameters
    HISTORY = 500               # 배경 모델링에 사용할 과거 프레임 수
    VAR_THRESHOLD = 50          # 임계값 (높을수록 미세한 움직임 무시)
    DETECT_SHADOWS = True       # 그림자 감지 여부 (회색으로 표시됨)
    
    # Morphological Params (Noise Removal)
    KERNEL_SIZE = 5
    
    # Blob Classification (Area in pixels)
    MIN_AREA = 800              # 최소 크기 (이보다 작으면 노이즈)
    LARGE_AREA = 10000          # 이보다 크면 차량 혹은 다수 그룹으로 간주
    
    # Tracking
    MAX_DISAPPEAR = 5           # 객체가 사라져도 ID 유지하는 프레임 수
    MAX_DISTANCE = 150          # 프레임 간 같은 객체로 간주할 최대 거리

class Blob:
    def __init__(self, blob_id, centroid, rect):
        self.id = blob_id
        self.centroid = centroid # (cx, cy)
        self.rect = rect         # (x, y, w, h)
        self.history = [centroid] # Path history
        self.disappeared_frames = 0
        self.speed = 0.0
        self.classification = "Unknown"

    def update(self, new_centroid, new_rect):
        # Calculate Speed (Distance moved per frame)
        dx = new_centroid[0] - self.centroid[0]
        dy = new_centroid[1] - self.centroid[1]
        self.speed = math.sqrt(dx**2 + dy**2)
        
        self.centroid = new_centroid
        self.rect = new_rect
        self.history.append(new_centroid)
        if len(self.history) > 20: self.history.pop(0)
        self.disappeared_frames = 0
        
        # Classify based on Size & Speed
        area = new_rect[2] * new_rect[3]
        if area > Config.LARGE_AREA:
            self.classification = "Group/Vehicle"
        elif self.speed > 15:
            self.classification = "Running"
        elif self.speed > 2:
            self.classification = "Walking"
        else:
            self.classification = "Static"

class Tracker:
    def __init__(self):
        self.next_id = 0
        self.blobs = {} # {id: BlobObject}

    def update(self, detected_rects):
        """
        Matches detected bounding boxes to existing blobs based on Euclidean distance.
        Simple Centroid Tracking algorithm.
        """
        # Calculate centroids from rects
        input_centroids = []
        for (x, y, w, h) in detected_rects:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids.append((cx, cy))

        # If no blobs exist, register all
        if len(self.blobs) == 0:
            for i in range(len(detected_rects)):
                self.register(input_centroids[i], detected_rects[i])
            return self.blobs

        # Match existing blobs to new centroids
        object_ids = list(self.blobs.keys())
        object_centroids = [b.centroid for b in self.blobs.values()]
        
        # Distance Matrix
        D = np.zeros((len(object_ids), len(input_centroids)))
        for i in range(len(object_ids)):
            for j in range(len(input_centroids)):
                dist = math.sqrt((object_centroids[i][0] - input_centroids[j][0])**2 + 
                                 (object_centroids[i][1] - input_centroids[j][1])**2)
                D[i, j] = dist

        # Find smallest distance pairs
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            if D[row, col] > Config.MAX_DISTANCE: continue

            obj_id = object_ids[row]
            self.blobs[obj_id].update(input_centroids[col], detected_rects[col])
            
            used_rows.add(row)
            used_cols.add(col)

        # Register new objects
        for col in range(len(input_centroids)):
            if col not in used_cols:
                self.register(input_centroids[col], detected_rects[col])

        # Deregister disappearing objects
        for row in range(len(object_ids)):
            if row not in used_rows:
                obj_id = object_ids[row]
                self.blobs[obj_id].disappeared_frames += 1
                if self.blobs[obj_id].disappeared_frames > Config.MAX_DISAPPEAR:
                    del self.blobs[obj_id]
                    
        return self.blobs

    def register(self, centroid, rect):
        self.blobs[self.next_id] = Blob(self.next_id, centroid, rect)
        self.next_id += 1

class BGSubDetector:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=Config.HISTORY, 
            varThreshold=Config.VAR_THRESHOLD, 
            detectShadows=Config.DETECT_SHADOWS
        )
        self.tracker = Tracker()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.KERNEL_SIZE, Config.KERNEL_SIZE))

    def run(self):
        print("[Info] Starting MOG2 Blob Tracking...")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (960, 540))
            
            # 1. Background Subtraction
            mask = self.bg_subtractor.apply(frame)
            
            # 2. Morphological Operations (Clean Noise)
            # Remove shadows (gray pixels in MOG2) -> Binary
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)
            
            # 3. Find Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rects = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > Config.MIN_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    rects.append((x, y, w, h))
            
            # 4. Update Tracker
            blobs = self.tracker.update(rects)
            
            # 5. Visualization
            vis_frame = frame.copy()
            for obj_id, blob in blobs.items():
                x, y, w, h = blob.rect
                
                # Color based on classification
                color = (0, 255, 0)
                if blob.classification == "Running": color = (0, 0, 255)
                elif blob.classification == "Group/Vehicle": color = (255, 0, 0)
                
                cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
                
                text = f"ID:{obj_id} {blob.classification} (v={blob.speed:.1f})"
                cv2.putText(vis_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw path
                if len(blob.history) > 1:
                    pts = np.array(blob.history, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_frame, [pts], False, color, 2)

            # Show Mask and Frame
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((vis_frame, mask_bgr))
            cv2.putText(combined, "Left: Tracking Result | Right: Foreground Mask", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("MOG2 Blob Analysis", combined)
            
            if cv2.waitKey(1) == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    src = "video1.mp4" if os.path.exists("video1.mp4") else 0
    detector = BGSubDetector(src)
    detector.run()