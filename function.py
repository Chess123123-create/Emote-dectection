# ===============================================
# function.py (update ổn định)
# -----------------------------------------------
# Nhận diện cảm xúc (real-time + ảnh tĩnh)
# Đã fix:
#   - Nhận diện nhiều mặt trong ảnh tĩnh
#   - Vẽ khung xanh cho khuôn mặt lớn nhất/rõ nhất
#   - Giữ smoothing + hysteresis
# ===============================================

import threading
import time
import base64
import cv2
from fer.fer import FER
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# -------------------------------------------------
# Khởi tạo mô hình FER
# -------------------------------------------------
_DETECTOR = None
def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = FER()
    return _DETECTOR

# -------------------------------------------------
# Quotes theo cảm xúc
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
# Frame -> PNG base64
# -------------------------------------------------
def frame_to_base64_png(frame_bgr):
    success, buf = cv2.imencode(".png", frame_bgr)
    if not success:
        raise ValueError("Failed to encode frame to PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# -------------------------------------------------
# Nhận diện cảm xúc từ frame
# -------------------------------------------------
def detect_emotion_from_frame(frame_bgr, detector=None, debug=False):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detector = detector or _get_detector()
    try:
        results = detector.detect_emotions(frame_rgb)
    except Exception as e:
        if debug:
            print(f"[ERROR] detect_emotions failed: {e}")
        return ("neutral", 0.0, [], {})

    if not results:
        return ("neutral", 0.0, [], {})

    def emotion_strength(r):
        em = r.get("emotions", {})
        return max(em.values()) if em else 0.0

    best = max(results, key=emotion_strength)
    emotions = best.get("emotions", {})
    if emotions:
        name, score = max(emotions.items(), key=lambda kv: kv[1])
    else:
        name, score = "neutral", 0.0

    boxes = [tuple(r["box"]) for r in results]

    # Vẽ khung: xanh cho best, đỏ cho các mặt khác
    for r, (x, y, w, h) in zip(results, boxes):
        color = (0, 255, 0) if r is best else (0, 0, 255)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

    if debug:
        bx, by, bw, bh = best["box"]
        print(f"[INFO] Face at ({bx},{by},{bw},{bh}) -> Emotion: {name} ({score:.2f})")

    return (name, float(score), boxes, emotions)

# -------------------------------------------------
# CameraStreamer (giữ smoothing + hysteresis)
# -------------------------------------------------
class CameraStreamer:
    def __init__(self, camera_index=0, callback=None, fps=10,
                 smooth_window=5, hysteresis_delta=0.15):
        self.camera_index = camera_index
        self.callback = callback
        self.fps = fps
        self._running = False
        self._thread = None
        self.cap = None
        self.smooth_window = smooth_window
        self.hysteresis_delta = hysteresis_delta
        self._emotion_history = []
        self._last_emotion = ("neutral", 0.0)

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

                # === Smoothing ===
                self._emotion_history.append((emotion, score))
                if len(self._emotion_history) > self.smooth_window:
                    self._emotion_history.pop(0)

                counts = {}
                for e, s in self._emotion_history:
                    counts[e] = counts.get(e, 0) + s
                stable_emotion = max(counts.items(), key=lambda kv: kv[1])[0]
                stable_score = counts[stable_emotion] / len(self._emotion_history)

                # === Hysteresis ===
                prev_name, prev_score = self._last_emotion
                if stable_emotion != prev_name and stable_score < prev_score + self.hysteresis_delta:
                    stable_emotion = prev_name
                    stable_score = prev_score

                emotion, score = stable_emotion, stable_score
                self._last_emotion = (emotion, score)

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
# Nhận diện cảm xúc từ ảnh tĩnh (fix chọn best face)
# -------------------------------------------------
def detect_emotion_from_image_path(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không mở được ảnh: {path}")

    emotion, score, boxes, emotions = detect_emotion_from_frame(img)

    if not boxes:
        return img, emotion, score, boxes, emotions

    # Chọn khuôn mặt lớn nhất
    best_box = max(boxes, key=lambda b: b[2] * b[3])

    # Vẽ khung xanh cho best, đỏ cho các mặt khác
    for (x, y, w, h) in boxes:
        color = (0, 255, 0) if (x, y, w, h) == best_box else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return img, emotion, score, boxes, emotions

# -------------------------------------------------
# Quote theo cảm xúc
# -------------------------------------------------
def get_quote_for_emotion(emotion_name):
    return QUOTES.get(emotion_name, QUOTES["neutral"])
