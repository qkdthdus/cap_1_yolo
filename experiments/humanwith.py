"""
Object Interaction & Abandonment Detector using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Optional

# =============================================================================
# Configuration
# =============================================================================
class Config:
    MODEL_PATH = 'yolov8n.pt'
    
    # Target Classes (COCO Index)
    # 0: person, 24: backpack, 26: handbag, 28: suitcase, 39: bottle, 67: cell phone
    TARGET_OBJECTS = [24, 26, 28, 67] 
    CLASS_NAMES = {0: 'Person', 24: 'Backpack', 26: 'Handbag', 28: 'Suitcase', 67: 'Phone'}
    
    # Thresholds
    INTERACTION_DIST = 150  # Pixel distance to consider 'ownership'
    ABANDON_TIME = 5.0      # Seconds before an unattended object is marked 'Abandoned'
    WARNING_TIME = 2.0      # Seconds before warning
    
    # Visuals
    COLOR_OWNED = (0, 255, 0)
    COLOR_WARNING = (0, 255, 255)
    COLOR_DANGER = (0, 0, 255)

# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class DetectedEntity:
    id: int # Tracking ID
    cls: int
    box: list # [x1, y1, x2, y2]
    centroid: tuple # (cx, cy)
    last_seen: float

class ObjectState:
    """
    Manages the state of a specific object (e.g., a specific backpack)
    """
    def __init__(self, obj_id, cls_name):
        self.obj_id = obj_id
        self.cls_name = cls_name
        self.owner_id: Optional[int] = None
        self.status = "Unknown" # Unknown, Owned, Dropped, Abandoned
        self.drop_time: Optional[float] = None
        
    def update_owner(self, person_id):
        self.owner_id = person_id
        self.status = "Owned"
        self.drop_time = None
        
    def mark_dropped(self):
        if self.status == "Owned":
            self.status = "Dropped"
            self.drop_time = time.time()
            
    def check_abandonment(self):
        if self.status in ["Dropped", "Warning"] and self.drop_time:
            elapsed = time.time() - self.drop_time
            if elapsed > Config.ABANDON_TIME:
                self.status = "ABANDONED"
            elif elapsed > Config.WARNING_TIME:
                self.status = "Warning"
                
    def get_color(self):
        if self.status == "Owned": return Config.COLOR_OWNED
        if self.status == "Warning": return Config.COLOR_WARNING
        if self.status == "ABANDONED": return Config.COLOR_DANGER
        return (200, 200, 200)

# =============================================================================
# Interaction Manager
# =============================================================================
class InteractionManager:
    def __init__(self):
        self.object_states: Dict[int, ObjectState] = {} # Key: Object Track ID
        
    def update(self, persons: List[DetectedEntity], objects: List[DetectedEntity]):
        """
        Core logic to match persons to objects
        """
        # 1. Cleanup old objects
        active_obj_ids = [o.id for o in objects]
        self.object_states = {k: v for k, v in self.object_states.items() if k in active_obj_ids}
        
        # 2. Register new objects
        for obj in objects:
            if obj.id not in self.object_states:
                self.object_states[obj.id] = ObjectState(obj.id, Config.CLASS_NAMES.get(obj.cls, "Obj"))

        # 3. Distance Matrix Calculation (Persons vs Objects)
        if not persons or not objects:
            # If no persons, all dropped objects might become abandoned
            for state in self.object_states.values():
                if state.status == "Owned":
                    state.mark_dropped()
                state.check_abandonment()
            return

        p_centroids = np.array([p.centroid for p in persons])
        o_centroids = np.array([o.centroid for o in objects])
        
        # Matrix shape: (num_persons, num_objects)
        # Using broadcasting to calculate euclidean distance matrix
        # dist_matrix[i][j] = distance between person i and object j
        diff = p_centroids[:, np.newaxis, :] - o_centroids[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        
        # 4. Determine Ownership & Interaction
        for j, obj in enumerate(objects):
            state = self.object_states[obj.id]
            
            # Find closest person
            closest_p_idx = np.argmin(dist_matrix[:, j])
            min_dist = dist_matrix[closest_p_idx, j]
            
            if min_dist < Config.INTERACTION_DIST:
                # Connected to a person
                person_id = persons[closest_p_idx].id
                state.update_owner(person_id)
            else:
                # Far from any person
                if state.status == "Owned":
                    state.mark_dropped()
                state.check_abandonment()

# =============================================================================
# Detector Engine
# =============================================================================
class InteractionDetector:
    def __init__(self, source):
        print(f"Loading Model for Interaction Detection...")
        self.model = YOLO(Config.MODEL_PATH)
        self.cap = cv2.VideoCapture(source)
        self.manager = InteractionManager()
        
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # Use 'track' to get IDs. We need IDs for both persons and objects.
            # Classes: 0(person) + TARGET_OBJECTS
            classes_to_detect = [0] + Config.TARGET_OBJECTS
            results = self.model.track(frame, persist=True, verbose=False, classes=classes_to_detect, tracker="bytetrack.yaml")
            
            persons = []
            objects = []
            
            # Parse Results
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                clss = results[0].boxes.cls.int().cpu().numpy()
                
                for box, track_id, cls in zip(boxes, ids, clss):
                    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                    entity = DetectedEntity(
                        id=track_id, cls=cls, box=box, centroid=(cx, cy), last_seen=time.time()
                    )
                    
                    if cls == 0:
                        persons.append(entity)
                    elif cls in Config.TARGET_OBJECTS:
                        objects.append(entity)

            # Update Logic
            self.manager.update(persons, objects)
            
            # Visualization
            self.draw_debug(frame, persons, objects)
            
            cv2.imshow("Interaction & Abandonment Detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        
    def draw_debug(self, frame, persons, objects):
        # Draw Persons
        for p in persons:
            cv2.rectangle(frame, (int(p.box[0]), int(p.box[1])), (int(p.box[2]), int(p.box[3])), (100, 100, 100), 1)
            cv2.putText(frame, f"P{p.id}", (int(p.box[0]), int(p.box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
        # Draw Objects & Lines
        for o in objects:
            if o.id not in self.manager.object_states: continue
            state = self.manager.object_states[o.id]
            color = state.get_color()
            
            # Box
            cv2.rectangle(frame, (int(o.box[0]), int(o.box[1])), (int(o.box[2]), int(o.box[3])), color, 2)
            
            # Status Text
            label = f"{state.cls_name} [{state.status}]"
            if state.owner_id is not None and state.status == "Owned":
                label += f" (P{state.owner_id})"
                
                # Draw Line to Owner
                owner_entity = next((p for p in persons if p.id == state.owner_id), None)
                if owner_entity:
                    cv2.line(frame, o.centroid, owner_entity.centroid, color, 2)
            
            cv2.putText(frame, label, (int(o.box[0]), int(o.box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
    import os
    target_source = "video1.mp4" if os.path.exists("video1.mp4") else 0
    detector = InteractionDetector(target_source)
    detector.run()