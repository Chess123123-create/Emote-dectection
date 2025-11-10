# ===============================================
# function.py
# -----------------------------------------------
# File này chứa các hàm và lớp phục vụ cho việc:
#   - Nhận diện cảm xúc trên ảnh hoặc camera.
#   - Chuyển đổi ảnh sang định dạng phù hợp cho giao diện (base64 PNG).
#   - Cung cấp các câu trích dẫn vui tương ứng với cảm xúc.
# ===============================================

import threading      # Dùng để tạo luồng chạy song song (xử lý camera mà không chặn giao diện).
import time           # Dùng để đo thời gian, kiểm soát tốc độ khung hình (FPS).
import base64         # Dùng để mã hóa ảnh sang base64 (chuẩn truyền dữ liệu trên web/app).
import cv2            # Thư viện OpenCV – xử lý ảnh, video, camera.
from fer.fer import FER    # Thư viện FER – phát hiện khuôn mặt và nhận diện cảm xúc.
import numpy as np    # Thư viện toán học – hỗ trợ xử lý ảnh, ma trận.

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
# Hàm: frame_to_base64_png
# Mục đích: Chuyển ảnh BGR (chuẩn OpenCV) sang base64 định dạng PNG
# để hiển thị hoặc truyền qua mạng.
# -------------------------------------------------
def frame_to_base64_png(frame_bgr):
    """Convert khung hình OpenCV (BGR) -> chuỗi base64 định dạng PNG."""
    _, buf = cv2.imencode(".png", frame_bgr)  # Mã hóa ảnh thành định dạng PNG.
    b = buf.tobytes()                         # Chuyển buffer sang bytes.
    return base64.b64encode(b).decode("utf-8")  # Mã hóa sang base64 và trả về chuỗi UTF-8.

# -------------------------------------------------
# Hàm: detect_emotion_from_frame
# Mục đích: Nhận diện cảm xúc từ 1 khung hình (ảnh BGR).
# Kết quả trả về: (emotion_name, confidence_score, face_boxes)
# -------------------------------------------------
def detect_emotion_from_frame(frame_bgr):
    """
    Input:
        - frame_bgr: ảnh khung hình (BGR - chuẩn của OpenCV)
    Output:
        - emotion_name: tên cảm xúc có xác suất cao nhất
        - confidence: độ tin cậy (0.0 - 1.0)
        - face_boxes: danh sách tọa độ các khuôn mặt
        - emotions_dict: dict chứa xác suất từng loại cảm xúc
    """
    # 1️⃣ Chuyển sang RGB (FER yêu cầu ảnh RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 2️⃣ Dò cảm xúc bằng FER
    try:
        results = _detector.detect_emotions(frame_rgb)
    except Exception as e:
        print(f"[ERROR] detect_emotions failed: {e}")
        return ("neutral", 0.0, [], {})

    if not results:
        return ("neutral", 0.0, [], {})

    # 3️⃣ Ưu tiên khuôn mặt có cảm xúc mạnh nhất, thay vì chỉ lớn nhất theo diện tích
    def emotion_strength(r):
        """Tính cảm xúc mạnh nhất của 1 khuôn mặt"""
        em = r.get("emotions", {})
        return max(em.values()) if em else 0.0

    best = max(results, key=lambda r: emotion_strength(r))
    emotions = best.get("emotions", {})

    # 4️⃣ Lấy tên và điểm cảm xúc cao nhất
    if emotions:
        name, score = max(emotions.items(), key=lambda kv: kv[1])
    else:
        name, score = "neutral", 0.0

    # 5️⃣ Vẽ khung khuôn mặt – đỏ cho “best”, xanh cho các khuôn mặt khác
    for r in results:
        x, y, w, h = r["box"]
        if r is best:
            color = (0, 0, 255)  # Đỏ
        else:
            color = (0, 255, 0)  # Xanh lá
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

    # 6️⃣ Log debug ra console
    bx, by, bw, bh = best["box"]
    print(f"[INFO] Face at ({bx},{by},{bw},{bh}) -> Emotion: {name} ({score:.2f})")

    return (name, float(score), [best["box"]], emotions)

# -------------------------------------------------
# Lớp: CameraStreamer
# Mục đích: Mở webcam và chạy nhận diện cảm xúc song song (đa luồng).
# Callback được gọi mỗi khi có khung hình mới.
# -------------------------------------------------
class CameraStreamer:
    """
    Camera streamer chạy song song với luồng thread và trả về: (frame_bgr, emotion, score, boxes)
    callback signature: fn(frame_bgr, emotion_name, score, boxes)
    """
    def __init__(self, camera_index=0, callback=None, fps=10):
        self.camera_index = camera_index
        self.callback = callback
        self.fps = fps
        self._running = False
        self._thread = None
        self.cap = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

    def _run(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            time.sleep(0.2)
            interval = 1.0 / max(1, self.fps)
            while self._running:
                t0 = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                try:
                    emotion, score, boxes, emotions = detect_emotion_from_frame(frame)
                except Exception:
                    emotion, score, boxes, emotions = "neutral", 0.0, [], {}
                if self.callback:
                    self.callback(frame, emotion, score, boxes)
                dt = time.time() - t0
                to_sleep = interval - dt
                if to_sleep > 0:
                    time.sleep(to_sleep)
        finally:
            if self.cap:
                self.cap.release()

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
# Hàm: get_quote_for_emotion
# Mục đích: Lấy câu quote phù hợp với cảm xúc hiện tại.
# -------------------------------------------------
def get_quote_for_emotion(emotion_name):
    return QUOTES.get(emotion_name, QUOTES["neutral"])