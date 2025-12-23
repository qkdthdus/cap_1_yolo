import cv2

class ScreenUnit:
    def __init__(self, name, video_path, init_size=(640, 360)):
        self.name = name
        self.video_path = video_path
        self.active = False
        self.cap = None

        # ▶ 첫 프레임 로드
        tmp = cv2.VideoCapture(self.video_path)
        ret, frame = tmp.read()
        tmp.release()

        if not ret:
            raise RuntimeError(f"비디오 로드 실패: {video_path}")

        self.first_frame = frame
        self.init_w, self.init_h = init_size

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.init_w, self.init_h)

    def play(self):
        if self.active:
            return
        self.cap = cv2.VideoCapture(self.video_path)
        self.active = True

    def update(self):
        # ▶ 재생 중
        if self.active:
            ret, frame = self.cap.read()
            if not ret:
                # 끝나면 정지 상태로 복귀
                self.cap.release()
                self.active = False
                frame = self.first_frame
        else:
            # ▶ 정지 상태 → 첫 프레임 유지
            frame = self.first_frame

        # ▶ 항상 화면에 그림
        w = cv2.getWindowImageRect(self.name)[2]
        h = cv2.getWindowImageRect(self.name)[3]
        if w > 0 and h > 0:
            frame = cv2.resize(frame, (w, h))

        cv2.imshow(self.name, frame)
