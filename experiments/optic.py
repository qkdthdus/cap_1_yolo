"""
Optical Flow Anomaly Detector
"""

import cv2
import numpy as np
import time

class Config:
    # Input Resolution (Processing at lower res improves FPS)
    PROCESS_WIDTH = 640
    
    # Optical Flow Parameters (Farneback)
    PYR_SCALE = 0.5
    LEVELS = 3
    WINSZIE = 15
    ITERATIONS = 3
    POLY_N = 5
    POLY_SIGMA = 1.2
    
    # Thresholds
    MOTION_THRESHOLD = 2.0      # Minimum vector magnitude to consider as "movement"
    PANIC_THRESHOLD = 15.0      # Average magnitude to trigger "Panic/Abnormal" alert
    SMOOTHING = 0.1             # Exponential moving average factor for stable readings

class Visualizer:
    @staticmethod
    def flow_to_hsv(flow):
        """
        Converts flow vectors (dx, dy) to BGR image for visualization using HSV color space.
        Angle -> Hue, Magnitude -> Value
        """
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255 # Saturation max
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Angle to Hue (0~180 in OpenCV)
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # Magnitude to Value (Normalized for visibility)
        # Cap magnitude at 20 for visualization purposes
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), mag

class OpticalFlowDetector:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Video source invalid.")
            
        self.prev_gray = None
        self.avg_energy = 0.0 # To store smoothed energy level
        
    def run(self):
        print("[Info] Starting Optical Flow Analysis...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # 1. Preprocessing
            # Resize for performance
            height, width = frame.shape[:2]
            scale = Config.PROCESS_WIDTH / width
            dim = (Config.PROCESS_WIDTH, int(height * scale))
            
            frame_resized = cv2.resize(frame, dim)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            if self.prev_gray is None:
                self.prev_gray = gray
                continue
                
            # 2. Calculate Dense Optical Flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                Config.PYR_SCALE, Config.LEVELS, Config.WINSZIE,
                Config.ITERATIONS, Config.POLY_N, Config.POLY_SIGMA, 0
            )
            
            # 3. Analyze Flow Vectors
            # Visualize flow and get magnitude matrix
            flow_vis, mag = Visualizer.flow_to_hsv(flow)
            
            # Calculate Scene Energy (Average Motion Magnitude)
            # Filter out small noise (< MOTION_THRESHOLD)
            valid_motion = mag[mag > Config.MOTION_THRESHOLD]
            
            current_energy = np.mean(valid_motion) if len(valid_motion) > 0 else 0
            
            # Smooth the energy value
            self.avg_energy = (self.avg_energy * (1 - Config.SMOOTHING)) + (current_energy * Config.SMOOTHING)
            
            # 4. Status Determination
            status = "Normal"
            status_color = (0, 255, 0)
            
            if self.avg_energy > Config.PANIC_THRESHOLD:
                status = "!!! PANIC / HIGH ACTIVITY !!!"
                status_color = (0, 0, 255)
            elif self.avg_energy > Config.PANIC_THRESHOLD * 0.4:
                status = "Active Movement"
                status_color = (0, 165, 255) # Orange
                
            # 5. Visualization Composition
            # Original + Flow Visualization Side by Side
            combined = np.hstack((frame_resized, flow_vis))
            
            # Overlay Info
            cv2.rectangle(combined, (0, 0), (combined.shape[1], 60), (0,0,0), -1)
            
            # Energy Bar
            bar_width = int((self.avg_energy / Config.PANIC_THRESHOLD) * 300)
            cv2.rectangle(combined, (20, 40), (20 + bar_width, 50), status_color, -1)
            cv2.rectangle(combined, (20, 40), (320, 50), (255,255,255), 1)
            
            text = f"Status: {status} | Energy: {self.avg_energy:.2f}"
            cv2.putText(combined, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow("Optical Flow Analysis", combined)
            
            self.prev_gray = gray
            
            if cv2.waitKey(1) == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    # 파일이 없으면 웹캠(0) 사용
    src = "video1.mp4" if os.path.exists("video1.mp4") else 0
    detector = OpticalFlowDetector(src)
    detector.run()