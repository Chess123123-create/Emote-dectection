import os
import flet as ft
from function import (
    CameraStreamer,
    frame_to_base64_png,
    get_quote_for_emotion,
    detect_emotion_from_image_path,
)

# Tắt một số tối ưu hóa của TensorFlow/oneDNN để tránh hiện tượng crash/giảm hiệu năng trên một số máy.
# Một số người dùng gặp lỗi khi dùng onednn; thiết lập này là "biện pháp phòng" thường thấy.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class AppUI:
    """
    Lớp quản lý toàn bộ giao diện và luồng xử lý chính của ứng dụng.
    - page: đối tượng ft.Page được Flet truyền vào.
    - chịu trách nhiệm: khởi tạo UI, chuyển giữa các trang (start / camera / image result),
      khởi/dừng CameraStreamer, xử lý file picker, cập nhật UI khi có frame mới.
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Emotion Detector"
        self.page.window_width = 900
        self.page.window_height = 600
        self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.page.bgcolor = "#0f1724"

        # Trang thái mở rộng ảnh camera (khi click vào ảnh sẽ to ra)
        self.camera_expanded = False
        # Biến giữ đối tượng CameraStreamer (nếu đang mở camera)
        self.streamer = None

        # FILE PICKER
        # Dùng để chọn file ảnh từ máy người dùng cho chế độ "nhận diện qua ảnh".
        # on_result sẽ gọi _on_file_picked khi user chọn xong.
        self.file_picker = ft.FilePicker(on_result=self._on_file_picked)
        # FilePicker nằm trong overlay của page (ẩn mặc định, được mở khi gọi pick_files()).
        self.page.overlay.append(self.file_picker)

        # ---------- Biến UI chính (các control sẽ được thêm vào layout) ---------- 
        # Ảnh hiển thị khung camera (khi chạy real-time) hoặc ảnh kết quả khi nhận diện từ file.
        self.camera_image = ft.Image(
            src="images\camera-not-available.jpg" if os.path.exists("images\camera-not-available.jpg") else None,  # khởi tạo rỗng, sẽ gán src_base64 khi có frame
            width=360,
            height=270,
            fit=ft.ImageFit.CONTAIN,
            border_radius=ft.border_radius.all(16),
        )

        # Text hiển thị câu "quote" tương ứng với cảm xúc (vd: an ủi khi buồn, chọc cười khi vui)
        self.quote_text = ft.Text(
            get_quote_for_emotion("neutral"),  # mặc định là neutral lúc chưa có kết quả
            size=15,
            italic=True,
            color="#cbd5e1",
            text_align=ft.TextAlign.CENTER,
        )

        # Thanh text hiển thị tên cảm xúc + độ tin cậy
        self.emotion_bar = ft.Text(
            "Cảm xúc: --",
            size=20,
            weight=ft.FontWeight.W_600,
            color="#ffffff",
        )

        # Xây dựng trang mở đầu (Start Page)
        self.build_start_page()

    # -------------------- Start Page --------------------
    def build_start_page(self):
        """
        Xây dựng giao diện trang đầu:
        - Dừng streamer nếu đang chạy (tránh rò camera)
        - Hiển thị 2 nút: real-time và nhận diện qua ảnh
        """
        if self.streamer:
            # Nếu trước đó có streamer chạy thì dừng để giải phóng camera
            self.streamer.stop()

        # Dọn page hiện tại trước khi add các control mới
        self.page.clean()

        # Nút chuyển sang chế độ nhận diện real-time (webcam)
        btn_rt = ft.ElevatedButton(
            "Nhận diện cảm xúc real-time",
            on_click=self.on_rt_click,
            width=260,
            style=ft.ButtonStyle(
                bgcolor="#0ea5e9", color="#021025", shape=ft.RoundedRectangleBorder(8)
            ),
        )

        # Nút chuyển sang chế độ nhận diện qua ảnh (mở file picker)
        btn_img = ft.ElevatedButton(
            "Nhận diện qua ảnh",
            on_click=lambda _: self.file_picker.pick_files(allow_multiple=False),
            width=260,
            style=ft.ButtonStyle(
                bgcolor="#0369a1", color="#ffffff", shape=ft.RoundedRectangleBorder(8)
            ),
        )

        # Header chứa tiêu đề, mô tả và 2 nút chọn
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

        # Add header vào page và render
        self.page.add(header)
        self.page.update()

    # -------------------- Real-time flow --------------------
    def on_rt_click(self, e):
        """Xử lý khi người dùng bấm nút real-time."""
        self.show_camera_ui()

    def show_camera_ui(self):
        """
        Xây dựng giao diện camera (layout gồm ảnh camera + box quote + thanh cảm xúc)
        - Khởi tạo CameraStreamer để bắt đầu lấy frame liên tục.
        """
        # Dọn giao diện hiện tại
        self.page.clean()

        # Nút quay lại (trở về trang chính)
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

        # Row chứa nút quay lại (bên trái)
        back_row = ft.Row(
            [back_button],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

        # Main row: trái là container chứa camera image (có thể click để expand),
        # phải là column chứa tên "Chisa" và quote.
        main_row = ft.Row(
            [
                ft.Container(
                    # GestureDetector bọc camera_image để bắt sự kiện on_tap (click ảnh để to ra)
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

        # Thanh hiển thị cảm xúc (Container để có background và padding)
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

        # Tổ hợp layout chính của trang camera
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

        # Add layout vào page
        self.page.add(layout)

        # -------- Khởi động camera --------
        # Nếu streamer cũ còn đang chạy thì stop trước khi tạo streamer mới
        if self.streamer:
            self.streamer.stop()
        # Tạo CameraStreamer với callback on_new_frame, fps = 8
        self.streamer = CameraStreamer(callback=self.on_new_frame, fps=8)
        self.streamer.start()

        # Khi click page (không phải ảnh), có thể thu nhỏ ảnh nếu đang mở lớn
        self.page.on_click = self.on_page_click

        # Render page
        self.page.update()

    def toggle_camera_size(self, e):
        """
        Thay đổi kích cỡ camera_image khi người dùng click vào ảnh.
        - Nếu đang thu nhỏ (default) -> phóng to theo tỉ lệ cửa sổ.
        - Nếu đang phóng to -> thu nhỏ về kích thước mặc định.
        """
        self.camera_expanded = not self.camera_expanded
        if self.camera_expanded:
            # Phóng to gần hết cửa sổ
            self.camera_image.width = int(self.page.window_width * 0.9)
            self.camera_image.height = int(self.page.window_height * 0.7)
        else:
            # Trả về kích thước mặc định
            self.camera_image.width = 360
            self.camera_image.height = 270
        # Cập nhật UI
        self.page.update()

    def on_page_click(self, e):
        """
        Bắt event click trên page: nếu ảnh đang mở lớn (expanded) -> đóng lại.
        Đây là cách đơn giản để người dùng click ra vùng ngoài để thu nhỏ.
        """
        if self.camera_expanded:
            self.camera_expanded = False
            self.camera_image.width = 360
            self.camera_image.height = 270
            self.page.update()

    def on_new_frame(self, frame_bgr, emotion, score, boxes):
        """
        Callback được CameraStreamer gọi mỗi khi có frame mới.
        - frame_bgr: ảnh BGR (OpenCV)
        - emotion: tên cảm xúc (string)
        - score: độ tin cậy (float)
        - boxes: list chứa box khuôn mặt (x, y, w, h)
        Mục tiêu: chuyển frame -> base64 -> cập nhật image và text trên UI.
        """
        # Chuyển frame (BGR) sang base64 PNG để dùng trong Flet (src_base64).
        b64 = frame_to_base64_png(frame_bgr)

        # Đóng gói update UI vào hàm nội bộ để dễ gọi với invoke_later
        def update_ui():
            # Gán dữ liệu ảnh
            self.camera_image.src_base64 = b64
            # Cập nhật text cảm xúc + score (format 2 chữ số thập phân)
            self.emotion_bar.value = f"Cảm xúc: {emotion.upper()}  ({score:.2f})"
            # Cập nhật quote theo cảm xúc
            self.quote_text.value = get_quote_for_emotion(emotion)
            # Cập nhật page
            self.page.update()

        # Một số phiên bản Flet không có invoke_later -> dùng try/except
        # invoke_later hữu ích khi callback được gọi từ thread khác (ở đây CameraStreamer chạy thread)
        # invoke_later sẽ chạy update_ui trên main thread của Flet an toàn.
        try:
            self.page.invoke_later(update_ui)
        except AttributeError:
            # Nếu không có invoke_later (phiên bản Flet cũ) thì gọi trực tiếp.
            update_ui()

    def back_to_main(self):
        """Dừng stream (nếu có) rồi đưa về trang start."""
        if self.streamer:
            self.streamer.stop()
        self.build_start_page()

    # -------------------- Nhận diện qua ảnh --------------------
    def on_image_click(self, e):
        """Khi user chọn 'Nhận diện qua ảnh' -> mở file picker."""
        self.page.update()
        # Mở dialog chọn file (FilePicker đã được thêm vào overlay lúc init)
        self.file_picker.pick_files(allow_multiple=False)

    def _on_file_picked(self, e: ft.FilePickerResultEvent):
        """
        Callback khi người dùng đã chọn 1 file qua FilePicker.
        - Lấy file (e.files) -> đảm bảo lưu bytes ra path nếu không có path (vd: web uploads).
        - Sau đó gọi detect_emotion_from_image_path để nhận diện.
        - Hiển thị ảnh kết quả + emoji + score + quote.
        """
        # Nếu user hủy (không chọn file) -> nothing to do
        if not e.files:
            return

        # Lấy file đầu tiên (chỉ cho phép 1 file)
        pf = e.files[0]
        # pf.path có thể là None nếu file được upload từ web client; fallback sang tên file tại cwd
        path = pf.path or f"./{pf.name}"
        # Nếu không có path nhưng có bytes -> ghi bytes ra file tạm để OpenCV có thể đọc
        if not pf.path and pf.bytes:
            with open(path, "wb") as f:
                f.write(pf.bytes)
        # image_path là đường dẫn thực sự tới file ảnh
        image_path = pf.path or path

        # Gọi hàm detect - bọc try/except để catch lỗi (vd file không phải ảnh, lỗi thư viện...)
        try:
            img_bgr, emotion, score, boxes, all_emotions = detect_emotion_from_image_path(image_path)
            emotion_details = "\n".join([f"{k}: {v:.2f}" for k, v in all_emotions.items()])
        except Exception as ex:
            # Hiện snack bar báo lỗi cho user
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Lỗi khi nhận diện ảnh: {ex}"))
            self.page.snack_bar.open = True
            self.page.update()
            return

        # Chuyển ảnh kết quả sang base64 để hiển thị
        b64 = frame_to_base64_png(img_bgr)

        # Bản đồ emoji tương ứng với tên cảm xúc (để trang trí giao diện)
        emoji = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "fear": "😨",
            "surprise": "😲",
            "disgust": "🤢",
            "neutral": "😐",
        }.get(emotion, "🙂")

        # Tạo view kết quả (ảnh + tên cảm xúc + độ tin cậy + quote + nút quay lại
        result_view = ft.Row(
            [
                # Bên trái: Hiển thị hình ảnh
                ft.Container(
                    content=ft.Image(
                        src_base64=b64,
                        width=480,
                        height=360,
                        fit=ft.ImageFit.CONTAIN,
                    ),
                    border_radius=20,
                    shadow=ft.BoxShadow(blur_radius=20, color="#00000088"),
                    alignment=ft.alignment.center,
                ),

                # Bên phải: Hiển thị kết quả
                ft.Column(
                    [
                        ft.Text(f"{emoji}  {emotion.upper()}", size=28, weight=ft.FontWeight.BOLD, color="#e6eef8"),
                        ft.Text(get_quote_for_emotion(emotion), size=16, italic=True, color="#cbd5e1", text_align=ft.TextAlign.CENTER),
                        ft.Text(f"Độ tin cậy: {score:.2f}", size=16, weight=ft.FontWeight.BOLD, color="#93c5fd"),
                        ft.Text(
                            f"Chi tiết xác suất:\n{emotion_details}",
                            size=16,
                            color="#fcfca5",
                        ),
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
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                    spacing=20,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_AROUND,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # Center the result view in the page
        centered = ft.Container(
            content=result_view,
            alignment=ft.alignment.center,
            expand=True,
        )

        print("\n--- Emotion Probabilities ---")
        for k, v in all_emotions.items():
            print(f"{k:10s}: {v:.4f}")
        print("-----------------------------\n")


        # Show the result view
        self.page.clean()
        self.page.add(centered)
        self.page.update()

    def clean_up(self):
        """
        Hàm dọn dẹp được gọi khi app đóng (vd page.on_close = app.clean_up()).
        Dừng streamer nếu còn chạy để giải phóng camera.
        """
        if self.streamer:
            self.streamer.stop()
