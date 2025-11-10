# ===============================================
# function.py
# -----------------------------------------------
# File này chứa các hàm và lớp phục vụ cho việc:
#   - Nhận diện cảm xúc trên ảnh hoặc camera.
#   - Chuyển đổi ảnh sang định dạng phù hợp cho giao diện (base64 PNG).
#   - Cung cấp các câu trích dẫn vui tương ứng với cảm xúc.
# ===============================================

import threading            # Chạy các tác vụ song song
import time                 # Đo thời gian, kiểm soát tốc độ khung hình (FPS)
import base64               # Mã hóa ảnh thành chuỗi text (để truyền qua mạng hoặc hiển thị trên web/app)
import cv2                  # Thư viện OpenCV – xử lý ảnh, video, camera.
from fer.fer import FER     # Thư viện FER – phát hiện khuôn mặt và nhận diện cảm xúc.
import numpy as np          # Xử lý ma trận, ảnh, và các phép toán tính toán nhanh.

# -------------------------------------------------
# Khởi tạo mô hình nhận diện cảm xúc FER.
# mtcnn=False: tắt phát hiện khuôn mặt bằng MTCNN (vì nặng và chậm hơn).
# -------------------------------------------------
_detector = FER(mtcnn=True)

# -------------------------------------------------
# Tập hợp các câu phản hồi tương ứng với cảm xúc.
# -------------------------------------------------
QUOTES = {
    "angry": "Hơi nóng lên rồi... chillax bruh.",
    "disgust": "Có vẻ cậu đang hơi khó chịu. Cố gắng giữ bình tĩnh nào.",
    "fear": "Hít thở đều nào, mọi thứ sẽ ổn thôi.",
    "happy": "Ồ! Điều gì đã khiến cậu vui như vậy? :)",
    "sad": "Mọi chuyện rồi sẽ qua thôi. Có tớ ở đây với cậu mà!",
    "surprise": "Ồ! Điều gì làm cậu ngạc nhiên thế?",
    "neutral": "Just a chill guy hanging around here, huh?",
}

# -------------------------------------------------
# Hàm này giúp chuyển ảnh thành chuỗi text, giống như gói ảnh thành dạng tin nhắn để gửi đi
def frame_to_base64_png(frame_bgr):
    # Mã hóa ảnh (numpy array BGR) thành file ảnh theo định dạng PNG nằm trong bộ nhớ (trả về buffer).
    _, buf = cv2.imencode(".png", frame_bgr)
    b = buf.tobytes()                         # Chuyển buffer sang bytes.
    return base64.b64encode(b).decode("utf-8")  # Mã hóa sang base64 và trả về chuỗi UTF-8.

# -------------------------------------------------
# Hàm: get_quote_for_emotion
# Mục đích: Lấy câu quote phù hợp với cảm xúc hiện tại.
# -------------------------------------------------
def get_quote_for_emotion(emotion_name):
    return QUOTES.get(emotion_name, QUOTES["neutral"])

# -------------------------------------------------
# Hàm: detect_emotion_from_frame
# Mục đích: Nhận diện cảm xúc từ 1 khung hình (ảnh BGR).
# Kết quả trả về: (emotion_name, confidence_score, face_boxes)
# -------------------------------------------------
def detect_emotion_from_frame(frame_bgr):
    """
    Input: ảnh khung hình (BGR - chuẩn của OpenCV)
    Output: (emotion_name, confidence, face_boxes, emotions_dict)
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _detector.detect_emotions(frame_rgb)

    if not results:
        return ("neutral", 0.0, [], {})

    def area(box):
        x, y, w, h = box
        return w * h

    best = max(results, key=lambda r: area(r["box"]))
    emotions = best.get("emotions", {})

    if emotions:
        top = max(emotions.items(), key=lambda kv: kv[1])
        name, score = top[0], float(top[1])
    else:
        name, score = "neutral", 0.0

    # Vẽ khung cho tất cả khuôn mặt, highlight khuôn mặt 'best' bằng màu đỏ
    for r in results:
        x, y, w, h = r["box"]
        # so sánh nội dung, tránh dùng `is` để an toàn
        if r == best:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

    return (name, score, [best["box"]], emotions)

# -------------------------------------------------
# Hàm: detect_emotion_from_image_path
# Mục đích: Đọc ảnh tĩnh từ file và nhận diện cảm xúc.
# -------------------------------------------------
def detect_emotion_from_image_path(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không mở được ảnh: {path}")
    
    # Phát hiện cảm xúc và các khuôn mặt
    emotion, score, boxes, emotions = detect_emotion_from_frame(img)

    # === Vẽ khung xanh lá quanh tất cả khuôn mặt phát hiện được ===
    if boxes:
        for (x, y, w, h) in boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img, emotion, score, boxes, emotions

# -------------------------------------------------
# Lớp: CameraStreamer
# Mục đích: Mở webcam và chạy nhận diện cảm xúc song song (đa luồng).
# Callback được gọi mỗi khi có khung hình mới.
# -------------------------------------------------
class CameraStreamer:
    """
    Camera streamer chạy song song với luồng thread và trả về:
    (frame_bgr, emotion_name, score, boxes)
    callback signature: fn(frame_bgr, emotion_name, score, boxes)
    """

    def __init__(self, camera_index=0, callback=None, fps=10):
        self.camera_index = camera_index  # Số thứ tự camera (0 là mặc định).
        self.callback = callback          # Hàm xử lý kết quả mỗi khung hình.
        self.fps = fps                    # Số khung hình xử lý mỗi giây.
        self._running = False             # Trạng thái chạy/dừng.
        self._thread = None               # Thread xử lý nền.
        self.cap = None                   # Đối tượng camera OpenCV.

    # -------------------------------------------------
    # Bắt đầu luồng camera.
    # -------------------------------------------------
    def start(self):
        if self._running:
            return  # Nếu đang chạy rồi thì không làm lại.
        self._running = True
        # Tạo luồng chạy song song (daemon=True: tự dừng khi chương trình kết thúc).
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -------------------------------------------------
    # Dừng luồng camera.
    # -------------------------------------------------
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)  # Đợi thread kết thúc.
        if self.cap:
            try:
                self.cap.release()          # Giải phóng camera.
            except Exception:
                pass

    # -------------------------------------------------
    # Hàm chạy nền: đọc khung hình, nhận diện cảm xúc, gọi callback.
    # -------------------------------------------------
    def _run(self):
        try:
            # Mở camera (CAP_DSHOW dùng cho Windows để tránh cảnh báo).
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            time.sleep(0.2)                      # Đợi camera ổn định.
            interval = 1.0 / max(1, self.fps)    # Tính thời gian giữa 2 khung hình theo FPS.

            while self._running:                 # Vòng lặp chính.
                t0 = time.time()
                ret, frame = self.cap.read()      # Đọc 1 khung hình.
                if not ret:                       # Nếu lỗi, thử lại sau 0.1s.
                    time.sleep(0.1)
                    continue

            try:
                # Nhận diện cảm xúc trên khung hình.
                emotion, score, boxes, emotions = detect_emotion_from_frame(frame)
            except Exception:
                emotion, score, boxes = "neutral", 0.0, []

                # Nếu có callback, gọi để gửi dữ liệu ra ngoài (VD: cập nhật UI).
                if self.callback:
                    self.callback(frame, emotion, score, boxes)

                # Điều chỉnh tốc độ để đạt FPS mong muốn.
                dt = time.time() - t0
                to_sleep = interval - dt
                if to_sleep > 0:
                    time.sleep(to_sleep)

        finally:
            # Đảm bảo luôn giải phóng camera khi dừng.
            if self.cap:
                self.cap.release()
