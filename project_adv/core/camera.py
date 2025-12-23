import cv2

class Camera:
    def __init__(self, index=0, flip=True):
        self.cap = cv2.VideoCapture(index)
        self.flip = flip

        if not self.cap.isOpened():
            raise RuntimeError("웹캠 열기 실패")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.flip:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        self.cap.release()
