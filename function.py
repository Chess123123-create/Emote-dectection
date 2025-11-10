import threading
import time
import base64
import io
import cv2
from fer import FER
import numpy as np

_detector = FER(mtcnn=False)  

QUOTES = {
    "angry": "Hơi nóng lên rồi... chillax bruh.",
    "disgust": "Có vẻ cậu đang hơi khó chịu. Cố gắng giữ bình tĩnh nào.",
    "fear": "Hít thở đều nào, mọi thứ sẽ ổn thôi.",
    "happy": "Ồ! Điều gì đã khiến cậu vui như vậy? :)",
    "sad": "Mọi chuyện rồi sẽ qua thôi. Có tớ ở đây với cậu mà!",
    "surprise": "Ồ! Điều gì làm cậu ngạc nhiên thế?",
    "neutral": "Just a chill guy hanging around here, huh?",
}

def frame_to_base64_png(frame_bgr):
    """cái này là để convert khung hình của bgr opencv -> based64 png cho phù hợp với flet, nch là k cần qtam."""
    _, buf = cv2.imencode(".png", frame_bgr)
    b = buf.tobytes()
    return base64.b64encode(b).decode("utf-8")

def detect_emotion_from_frame(frame_bgr):
    """
    Input: cái bgr ban nãy
    Return: (top_emotion_name, confidence, face_boxes(list))
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _detector.detect_emotions(frame_rgb)
    if not results:
        return ("neutral", 0.0, [])

    def area(box):
        x, y, w, h = box
        return w * h
    best = max(results, key=lambda r: area(r["box"]))
    emotions = best.get("emotions", {})

    if not emotions:
        return ("neutral", 0.0, [best["box"]])
    top = max(emotions.items(), key=lambda kv: kv[1])
    name, score = top[0], float(top[1])
    return (name, score, [best["box"]])

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
                    emotion, score, boxes = detect_emotion_from_frame(frame)
                except Exception:
                    emotion, score, boxes = "neutral", 0.0, []
                if self.callback:
                    self.callback(frame, emotion, score, boxes)
                dt = time.time() - t0
                to_sleep = interval - dt
                if to_sleep > 0:
                    time.sleep(to_sleep)
        finally:
            if self.cap:
                self.cap.release()

def detect_emotion_from_image_path(path):
    """Tải ảnh và nhận diện rồi trả về: (image_bgr, emotion, score, boxes)"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không mở được ảnh: {path}")
    emotion, score, boxes = detect_emotion_from_frame(img)
    return img, emotion, score, boxes

def get_quote_for_emotion(emotion_name):
    return QUOTES.get(emotion_name, QUOTES["neutral"])