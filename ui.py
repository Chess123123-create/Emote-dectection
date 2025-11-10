import os
import flet as ft
from function import (
    CameraStreamer,
    frame_to_base64_png,
    get_quote_for_emotion,
    detect_emotion_from_image_path,
)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class AppUI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Emotion Detector"
        self.page.window_width = 900
        self.page.window_height = 600
        self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.page.bgcolor = "#0f1724"

        self.camera_expanded = False
        self.streamer = None

        # file picker (Xóa đi là chatgpt cũng không cứu được đâu đó)
        self.file_picker = ft.FilePicker(on_result=self._on_file_picked)
        self.page.overlay.append(self.file_picker)

        # biến UI
        self.camera_image = ft.Image(
            src="",
            width=360,
            height=270,
            fit=ft.ImageFit.CONTAIN,
            border_radius=ft.border_radius.all(16),
        )
        self.quote_text = ft.Text(
            get_quote_for_emotion("neutral"),
            size=15,
            italic=True,
            color="#cbd5e1",
            text_align=ft.TextAlign.CENTER,
        )
        self.emotion_bar = ft.Text(
            "Cảm xúc: --",
            size=20,
            weight=ft.FontWeight.W_600,
            color="#ffffff",
        )

        self.build_start_page()

    # Trang mở đầu
    def build_start_page(self):
        if self.streamer:
            self.streamer.stop()

        self.page.clean()
        btn_rt = ft.ElevatedButton(
            "Nhận diện cảm xúc real-time",
            on_click=self.on_rt_click,
            width=260,
            style=ft.ButtonStyle(
                bgcolor="#0ea5e9", color="#021025", shape=ft.RoundedRectangleBorder(8)
            ),
        )
        btn_img = ft.ElevatedButton(
            "Nhận diện qua ảnh",
            on_click=self.on_image_click,
            width=260,
            style=ft.ButtonStyle(
                bgcolor="#0369a1", color="#ffffff", shape=ft.RoundedRectangleBorder(8)
            ),
        )

        header = ft.Column(
            [
                ft.Text("Emotion Detector Demo", size=28, weight=ft.FontWeight.BOLD, color="#e6eef8"),
                ft.Text(
                    "Chisa: trợ lý nhận diện cảm xúc riêng của bạn.",
                    size=13,
                    italic=True,
                    color="#93c5fd",
                ),
                ft.Row([btn_rt, btn_img], alignment=ft.MainAxisAlignment.CENTER, spacing=30),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )
        self.page.add(header)
        self.page.update()

    # Real-time
    def on_rt_click(self, e):
        self.show_camera_ui()

    def show_camera_ui(self):
        self.page.clean()
        back_button = ft.ElevatedButton(
            "← Quay lại",
            on_click=lambda e: self.back_to_main(),
            style=ft.ButtonStyle(
                bgcolor="#0ea5e9",
                color="#021025",
                shape=ft.RoundedRectangleBorder(8),
                padding=ft.Padding(12, 4, 12, 4),
            ),
        )

        back_row = ft.Row(
            [back_button],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

        main_row = ft.Row(
            [
                ft.Container(
                    content=ft.GestureDetector(content=self.camera_image, on_tap=self.toggle_camera_size),
                    padding=8,
                    border_radius=16,
                    bgcolor="#071022",
                    shadow=ft.BoxShadow(blur_radius=20, color="#00000088"),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Chisa:", size=14, color="#94a3b8"),
                            self.quote_text,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=10,
                    alignment=ft.alignment.center,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
            expand=True,
        )

        emotion_bar = ft.Container(
            content=ft.Row(
                [ft.Icon(name="favorite", color="#ffffff"), self.emotion_bar],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
            ),
            height=52,
            bgcolor="#0ea5e9",
            border_radius=ft.border_radius.all(12),
            margin=ft.Margin(200, 10, 200, 10),
            padding=ft.Padding(12, 8, 12, 8),
        )

        layout = ft.Column(
            [
                back_row,
                ft.Container(content=main_row, alignment=ft.alignment.center, expand=True),
                emotion_bar,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
            spacing=10,
        )

        self.page.add(layout)

        # Khởi động camera
        if self.streamer:
            self.streamer.stop()
        self.streamer = CameraStreamer(callback=self.on_new_frame, fps=8)
        self.streamer.start()
        self.page.on_click = self.on_page_click
        self.page.update()

    def toggle_camera_size(self, e):
        self.camera_expanded = not self.camera_expanded
        if self.camera_expanded:
            self.camera_image.width = int(self.page.window_width * 0.9)
            self.camera_image.height = int(self.page.window_height * 0.7)
        else:
            self.camera_image.width = 360
            self.camera_image.height = 270
        self.page.update()

    def on_page_click(self, e):
        if self.camera_expanded:
            self.camera_expanded = False
            self.camera_image.width = 360
            self.camera_image.height = 270
            self.page.update()

    def on_new_frame(self, frame_bgr, emotion, score, boxes):
        b64 = frame_to_base64_png(frame_bgr)

        def update_ui():
            self.camera_image.src_base64 = b64
            self.emotion_bar.value = f"Cảm xúc: {emotion.upper()}  ({score:.2f})"
            self.quote_text.value = get_quote_for_emotion(emotion)
            self.page.update()

        try:
            self.page.invoke_later(update_ui)
        except AttributeError:
            update_ui()

    def back_to_main(self):
        if self.streamer:
            self.streamer.stop()
        self.build_start_page()

    # Nhận diện qua ảnh
    def on_image_click(self, e):
        self.page.update()
        self.file_picker.pick_files(allow_multiple=False)

    def _on_file_picked(self, e: ft.FilePickerResultEvent):
        if not e.files:
            return

        pf = e.files[0]
        path = pf.path or f"./{pf.name}"
        if not pf.path and pf.bytes:
            with open(path, "wb") as f:
                f.write(pf.bytes)
        image_path = pf.path or path

        try:
            img_bgr, emotion, score, boxes = detect_emotion_from_image_path(image_path)
        except Exception as ex:
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Lỗi khi nhận diện ảnh: {ex}"))
            self.page.snack_bar.open = True
            self.page.update()
            return

        b64 = frame_to_base64_png(img_bgr)
        emoji = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "fear": "😨",
            "surprise": "😲",
            "disgust": "🤢",
            "neutral": "😐",
        }.get(emotion, "🙂")

        result_view = ft.Column(
            [
                ft.Container(
                    content=ft.Image(src_base64=b64, width=480, height=360, fit=ft.ImageFit.CONTAIN),
                    border_radius=20,
                    shadow=ft.BoxShadow(blur_radius=20, color="#00000088"),
                    alignment=ft.alignment.center,
                ),
                ft.Text(f"{emoji}  {emotion.upper()}", size=28, weight=ft.FontWeight.BOLD, color="#e6eef8"),
                ft.Text(f"Độ tin cậy: {score:.2f}", size=16, color="#93c5fd"),
                ft.Text(get_quote_for_emotion(emotion), size=15, italic=True, color="#cbd5e1", text_align=ft.TextAlign.CENTER),
                ft.ElevatedButton(
                    "← Quay lại",
                    on_click=lambda ev: self.build_start_page(),
                    style=ft.ButtonStyle(
                        bgcolor="#0ea5e9",
                        color="#021025",
                        shape=ft.RoundedRectangleBorder(8),
                        padding=ft.Padding(16, 8, 16, 8),
                    ),
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
            spacing=15,
        )

        centered = ft.Container(
            content=result_view,
            alignment=ft.alignment.center,
            expand=True,
        )

        self.page.clean()
        self.page.add(centered)
        self.page.update()

    def clean_up(self):
        if self.streamer:
            self.streamer.stop()