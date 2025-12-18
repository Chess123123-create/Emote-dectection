import base64  # [KỸ THUẬT] Mã hóa dữ liệu nhị phân (ảnh) thành chuỗi ký tự để hiển thị lên giao diện Web/Flet
import io      # Thư viện xử lý luồng dữ liệu vào/ra (Input/Output Stream)
import os      # Tương tác với hệ điều hành (tạo thư mục, đường dẫn file)
import threading # [KỸ THUẬT] Đa luồng: Giúp tách việc xử lý ảnh (nặng) ra khỏi việc vẽ giao diện (nhẹ) để App không bị đơ
import time    # Dùng để đo thời gian hoặc tạo độ trễ (sleep) giảm tải CPU
from datetime import datetime # Lấy thời gian thực để đặt tên file không trùng lặp
from typing import List, Tuple, Optional # Type Hinting: Giúp code rõ ràng, dễ debug hơn

import cv2     # [THỊ GIÁC MÁY TÍNH] OpenCV: Thư viện xử lý ảnh số 1 thế giới (Đọc camera, biến đổi ma trận ảnh)
import flet as ft # [GIAO DIỆN] Framework UI hiện đại
import numpy as np # [TOÁN HỌC] Thư viện xử lý ma trận. Máy tính "nhìn" ảnh là một ma trận số khổng lồ (Height x Width x Channels)
from fer.fer import FER # [TRÍ TUỆ NHÂN TẠO] Thư viện nhận diện cảm xúc tích hợp sẵn Deep Learning
from PIL import Image, ImageDraw, ImageFont # Pillow: Thư viện xử lý file ảnh bổ trợ

# -------------------------- 1. CẤU HÌNH & KHỞI TẠO AI -------------------------- #

STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

#  Khởi tạo bộ phát hiện khuôn mặt
# Tham số mtcnn=True: Sử dụng mạng nơ-ron MTCNN (Multi-task Cascaded Convolutional Networks).
# - MTCNN gồm 3 mạng con (P-Net, R-Net, O-Net) hoạt động tuần tự.
# - Ưu điểm: Chính xác hơn Haar Cascade truyền thống, tìm được mặt nghiêng, mặt bị che khuất một phần.
# - Nhược điểm: Chậm hơn Haar một chút, nhưng máy hiện đại xử lý tốt.
fer_detector = FER(mtcnn=True)


# -------------------------- 2. CÁC HÀM XỬ LÝ ẢNH (IMAGE PROCESSING) -------------------------- #

def frame_to_base64(frame: np.ndarray) -> str:
    """
    [XỬ LÝ ẢNH] Chuyển đổi Ma trận ảnh (OpenCV) -> Chuỗi Base64 (Web UI).
    Lý do: OpenCV xử lý ảnh dạng mảng số (numpy array), nhưng giao diện Flet (nền tảng Web)
    chỉ hiểu ảnh dạng chuỗi mã hóa hoặc đường dẫn URL.
    """
    if frame is None: return ""
    try:
        # [TỐI ƯU HIỆU NĂNG] Resize ảnh trước khi hiển thị
        # Ảnh gốc từ Camera có thể là HD/FullHD (rất nặng).
        # Ta thu nhỏ về chiều ngang 480px để truyền tải lên giao diện nhanh hơn, giảm độ trễ (Lag).
        # Hàm cv2.resize sử dụng thuật toán nội suy (Interpolation) để tính toán lại điểm ảnh.
        h, w = frame.shape[:2]
        if w > 480:
            scale = 480 / w
            frame = cv2.resize(frame, (480, int(h * scale)))
            
        # Nén ma trận ảnh thành định dạng PNG trong bộ nhớ đệm (RAM)
        _, buf = cv2.imencode(".png", frame)
        # Mã hóa thành chuỗi ASCII
        return base64.b64encode(buf).decode("utf-8")
    except: return ""


def bgr_and_rgb(frame):
    """
    [LÝ THUYẾT MÀU SẮC] Chuyển đổi không gian màu.
    - OpenCV mặc định đọc ảnh theo chuẩn BGR (Blue-Green-Red).
    - Mắt người và các Model AI (như FER) lại nhìn theo chuẩn RGB (Red-Green-Blue).
    - Nếu không chuyển đổi: Môi màu đỏ (Red) sẽ bị AI nhìn thành màu xanh dương (Blue) -> Nhận diện sai.
    """
    if frame is None:
        return None, None
    
    # Nếu ảnh đầu vào là ảnh xám (2 chiều: Cao x Rộng), ta giả lập thành 3 chiều để đồng bộ
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Ép kiểu dữ liệu về uint8 (số nguyên không dấu 0-255) - chuẩn của ảnh số
    frame = frame.astype(np.uint8)
    
    # Dùng hàm cvtColor (Convert Color) để đảo vị trí kênh màu
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Trả về cả 2: frame (để vẽ giao diện OpenCV), rgb (để AI phân tích)
    return frame, rgb


def analyze_frame(frame: np.ndarray) -> Tuple[np.ndarray, str, List[str], List[Tuple[int, int, int, int]], List[str]]:
    """
    [TRÁI TIM HỆ THỐNG] Hàm phân tích cảm xúc chính.
    Quy trình: Input Frame -> Tiền xử lý -> Detect khuôn mặt -> Phân loại cảm xúc -> Vẽ kết quả -> Output.
    """
    # Bước 1: Chuẩn hóa màu sắc
    bgr_for_draw, rgb_for_fer = bgr_and_rgb(frame)
    
    # Bước 2: Gọi thư viện FER để quét khuôn mặt và dự đoán
    # Hàm này trả về list các dictionary, mỗi dict chứa: box (tọa độ), emotions (điểm số các cảm xúc)
    results = fer_detector.detect_emotions(rgb_for_fer)
    
    label = "Không phát hiện"
    detail_lines = []
    annotated = bgr_for_draw.copy() # Tạo bản sao (Copy) để vẽ đè lên, giữ nguyên ảnh gốc sạch
    
    # [LỌC NHIỄU] Thiết lập ngưỡng diện tích
    # Nếu khuôn mặt nhỏ hơn 40x40 pixel -> Coi là nhiễu hoặc quá xa -> Bỏ qua.
    min_area = 40 * 40
    
    # [NGƯỠNG TIN CẬY]
    # Nếu AI dự đoán cảm xúc cao nhất mà dưới 25% -> Không tin -> Bỏ qua.
    conf_threshold = 0.25
    
    accepted_boxes: List[Tuple[int, int, int, int]] = []
    all_labels: List[str] = [] # Danh sách lưu tên cảm xúc để phục vụ việc click vào box

    for r in results:
        (x, y, w, h) = r["box"] # Lấy tọa độ góc trái trên (x,y) và kích thước (w,h)
        
        # Kiểm tra điều kiện diện tích
        if w * h < min_area:
            continue
            
        emotions = r["emotions"] # Lấy danh sách điểm số (VD: happy: 0.9, sad: 0.01...)
        
        # [THUẬT TOÁN] Tìm cảm xúc có điểm số lớn nhất (Argmax)
        top_emotion = max(emotions, key=emotions.get)
        top_score = emotions[top_emotion]
        
        # Kiểm tra độ tin cậy
        if top_score < conf_threshold:
            detail_lines.append(
                f"Bỏ qua box {x},{y},{w},{h} | độ tin cậy thấp {top_emotion}:{top_score:.2f}"
            )
            continue
        
        # Tạo nhãn hiển thị: Ví dụ "happy (0.95)"
        this_label = f"{top_emotion} ({top_score:.2f})"
        
        label = this_label # Cập nhật nhãn chung (lấy cái cuối cùng)
        accepted_boxes.append((x, y, w, h)) # Lưu tọa độ hợp lệ
        all_labels.append(this_label)       # Lưu tên cảm xúc vào danh sách
        
        # Lưu log chi tiết cho từng mặt
        detail_lines.append(
            f"Box {x},{y},{w},{h} | " +
            ", ".join(f"{k}:{v:.2f}" for k, v in emotions.items())
        )
        
        # [VẼ ĐỒ HỌA RASTER] Dùng OpenCV vẽ trực tiếp lên ma trận ảnh
        # Vẽ hình chữ nhật màu xanh lá (0, 255, 0) độ dày 2px
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Viết chữ lên ảnh (dùng font Hershey Simplex đơn giản, nhanh)
        cv2.putText(
            annotated,
            top_emotion,
            (x, y - 10), # Vị trí đặt chữ (trên đầu box)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,         # Cỡ chữ
            (0, 255, 0), # Màu chữ
            2,           # Độ dày nét chữ
        )

    if not results:
        detail_lines.append("Không phát hiện khuôn mặt.")

    return annotated, label, detail_lines, accepted_boxes, all_labels


def save_image_with_label(img_bgr, label, prefix):
    """Hàm lưu ảnh xuống ổ cứng."""
    # Tạo tên file duy nhất dựa trên thời gian (Timestamp) để tránh ghi đè
    filename = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.png"
    path = os.path.join(STORAGE_DIR, filename)
    
    # Chuyển hệ màu về RGB để thư viện PIL lưu đúng màu (vì OpenCV đang giữ BGR)
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_img.save(path)
    return path


# -------------------------- 3. XÂY DỰNG GIAO DIỆN (UI BUILDERS) -------------------------- #

def home_view(page: ft.Page) -> ft.View:
    """Màn hình chính (Dashboard)"""
    logo = ft.Container(
        width=240, padding=15, bgcolor=ft.Colors.BLUE_400,
        border_radius=12, border=ft.border.all(2, "black"),
        alignment=ft.alignment.center,
        content=ft.Row(
            [ft.Icon(ft.Icons.FACE, size=32, color="white"),
             ft.Text("Xinhtu", size=22, weight="bold", color="white")],
            spacing=10, alignment="center"),
    )

    # Hàm điều hướng (Routing)
    def go_live(_): page.go("/live")
    def go_photo(_): page.go("/photo")
    def go_storage(_): page.go("/storage")

    return ft.View(
        route="/",
        controls=[
            ft.Column([
                logo,
                ft.ElevatedButton("Nhận diện theo thời gian thực", on_click=go_live, height=60, width=300, style=ft.ButtonStyle(text_style=ft.TextStyle(size=20))),
                ft.ElevatedButton("Nhận diện 1 khung hình (ảnh)", on_click=go_photo, height=60, width=300, style=ft.ButtonStyle(text_style=ft.TextStyle(size=20))),
            ], alignment="center", horizontal_alignment="center", expand=True, spacing=30),
        ],
        floating_action_button=ft.FloatingActionButton(icon=ft.Icons.HISTORY, on_click=go_storage),
        vertical_alignment="center", horizontal_alignment="center",
    )


class LiveState:
    """
    [QUẢN LÝ TRẠNG THÁI] Dùng mẫu Singleton (đơn nhất) để kiểm soát Camera.
    Biến 'running' đóng vai trò công tắc nguồn.
    """
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

live_state = LiveState()


def live_view(page: ft.Page) -> ft.View:
    """
    Màn hình Camera Real-time.
    Sử dụng kỹ thuật Frame Skipping và Threading để tối ưu.
    """
    preview = ft.Image(width=480, height=360, fit=ft.ImageFit.CONTAIN)
    label_text = ft.Text("Chưa nhận diện", size=20, weight="bold")
    log_list = ft.ListView(expand=1, spacing=5, height=160)

    # Cấu hình BottomSheet (bảng thông tin trượt từ dưới lên)
    bottom_sheet = ft.BottomSheet(
        content=ft.Container(
            padding=10, height=220,
            content=ft.Column([ft.Text("Log cảm xúc"), ft.Container(content=log_list, expand=True, height=200, alignment=ft.alignment.top_center)], expand=True, scroll=ft.ScrollMode.ALWAYS)
        ),
        show_drag_handle=True, enable_drag=True, dismissible=True, is_scroll_controlled=True
    )
    page.overlay.clear()
    page.overlay.append(bottom_sheet)
    
    def open_sheet(_=None): bottom_sheet.open = True; page.update()
    bottom_sheet.on_dismiss = lambda _: page.update()
    
    peek_bar = ft.Container(
        height=30, width=180, bgcolor=ft.Colors.GREY_200, border_radius=12,
        alignment=ft.alignment.center, on_click=open_sheet,
        content=ft.Row([ft.Icon(ft.Icons.DRAG_HANDLE, size=18, color=ft.Colors.GREY_600), ft.Text("Kéo lên để xem log", size=12)], spacing=6, alignment="center")
    )

    # [HÀM XỬ LÝ LUỒNG VIDEO]
    def update_stream():
        # Mở kết nối tới Webcam (Index 0 thường là cam mặc định)
        live_state.cap = cv2.VideoCapture(0)
        
        # [TỐI ƯU 1] Giảm độ phân giải đầu vào xuống VGA (640x480)
        # Giúp giảm lượng pixel phải xử lý trên mỗi khung hình -> Tăng FPS.
        live_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        live_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not live_state.cap.isOpened():
            label_text.value = "Không mở được camera"; page.update(); return
            
        frame_count = 0
        # Bộ nhớ tạm để lưu kết quả của frame trước (dùng cho frame skipping)
        last_boxes = []
        last_label = ""
        
        # Vòng lặp vô hạn đọc camera
        while live_state.running:
            ret, frame = live_state.cap.read()
            if not ret: continue
            
            frame_count += 1
            
            # [TỐI ƯU 2] Kỹ thuật Frame Skipping (Nhảy cóc khung hình)
            # AI rất nặng, nếu chạy trên mọi frame (30FPS) sẽ làm CPU quá tải -> Lag.
            # Ta chỉ chạy AI trên frame thứ 0, 3, 6... (Mỗi 3 frame chạy 1 lần).
            if frame_count % 3 == 0:
                # Gọi AI phân tích
                annotated, lbl, details, boxes, _ = analyze_frame(frame)
                
                # Lưu kết quả vào bộ nhớ tạm
                last_boxes = boxes
                last_label = lbl
                
                # Cập nhật UI
                label_text.value = lbl
                timestamp = datetime.now().strftime('%H:%M:%S')
                if "Không phát hiện" not in lbl:
                    log_list.controls.append(ft.Text(f"{timestamp} - {lbl}", size=12))
                    # Xóa bớt log cũ nếu quá dài để tiết kiệm RAM
                    if len(log_list.controls) > 100: log_list.controls.pop(0)
            else:
                # Ở các frame bị bỏ qua (1, 2, 4, 5...), ta KHÔNG chạy AI.
                # Thay vào đó, ta lấy kết quả của frame trước vẽ lại lên frame hiện tại.
                # Điều này tạo cảm giác video mượt mà (30FPS) dù AI chỉ chạy 10FPS.
                annotated = frame.copy()
                for (x, y, w, h) in last_boxes:
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Lấy tên cảm xúc (bỏ phần điểm số trong ngoặc cho gọn)
                    short_lbl = last_label.split('(')[0] if '(' in last_label else last_label
                    cv2.putText(annotated, short_lbl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                label_text.value = last_label

            # Cập nhật ảnh lên giao diện
            preview.src_base64 = frame_to_base64(annotated)
            page.update()
            
            # Ngủ cực ngắn (10ms) để nhường tài nguyên CPU cho việc vẽ giao diện
            time.sleep(0.01)
            
        if live_state.cap: live_state.cap.release()

    def start_stream():
        if live_state.running: return
        live_state.running = True
        # Chạy hàm update_stream trong một luồng riêng (Daemon Thread)
        # Daemon Thread sẽ tự động tắt khi chương trình chính tắt.
        live_state.thread = threading.Thread(target=update_stream, daemon=True)
        live_state.thread.start()

    def stop_stream(): live_state.running = False
    def back(_): stop_stream(); page.go("/")

    start_stream()

    return ft.View(
        route="/live",
        controls=[
            ft.AppBar(title=ft.Text("Nhận diện thời gian thực"), leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=back)),
            ft.Row([ft.Container(preview, expand=True, border=ft.border.all(1, ft.Colors.GREY)), ft.Column([ft.Text("Cảm xúc"), label_text], expand=True)], expand=True),
            ft.Text("Kéo thanh dưới để xem log"), ft.Container(height=4), ft.Row([peek_bar], alignment="center"),
        ],
        vertical_alignment="start",
    )


def photo_view(page: ft.Page) -> ft.View:
    """Màn hình Phân tích Ảnh tĩnh"""
    preview = ft.Image(width=480, height=360, fit=ft.ImageFit.CONTAIN, gapless_playback=True)
    label_text = ft.Text("Chưa nhận diện", size=20, weight="bold")
    detail_text = ft.Text("")
    
    # Các biến lưu trữ trạng thái của ảnh đang xem
    analyzed_image: List[np.ndarray] = [None] # Ảnh gốc đã vẽ khung xanh
    photo_boxes: List[List[Tuple[int, int, int, int]]] = [[]] # Danh sách tọa độ các mặt
    face_labels: List[List[str]] = [[]] # Danh sách tên cảm xúc của từng mặt
    
    logs = ft.ListView(expand=True, spacing=4, height=200)

    # Bottom Sheet hiển thị chi tiết
    bottom_sheet = ft.BottomSheet(
        content=ft.Container(padding=10, height=220, content=ft.Column([ft.Text("Chi tiết"), logs], expand=True, scroll=ft.ScrollMode.ALWAYS)),
        show_drag_handle=True, enable_drag=True, dismissible=True, is_scroll_controlled=True
    )
    page.overlay.clear()
    page.overlay.append(bottom_sheet)
    
    def open_sheet(_=None): bottom_sheet.open = True; page.update()
    bottom_sheet.on_dismiss = lambda _: page.update()
    
    peek_bar = ft.Container(
        height=30, width=180, bgcolor=ft.Colors.GREY_200, border_radius=12,
        alignment=ft.alignment.center, on_click=open_sheet,
        content=ft.Row([ft.Icon(ft.Icons.DRAG_HANDLE, size=18, color=ft.Colors.GREY_600), ft.Text("Kéo lên để xem chi tiết", size=12)], spacing=6, alignment="center")
    )

    def back(_): page.go("/")

    # [SỰ KIỆN] Khi người dùng chọn file ảnh
    def on_file_result(e: ft.FilePickerResultEvent):
        if not e.files: return
        file_path = e.files[0].path
        
        # Đọc ảnh từ ổ cứng vào RAM
        frame = cv2.imread(file_path)
        if frame is None:
            label_text.value = "Không đọc được ảnh"; page.update(); return
            
        # Gọi hàm phân tích
        annotated, lbl, details, boxes, labels_list = analyze_frame(frame)
        
        # Lưu kết quả vào biến nhớ
        analyzed_image[0] = annotated
        photo_boxes[0] = boxes
        face_labels[0] = labels_list 
        
        label_text.value = lbl
        detail_text.value = "\n".join(details)

        # Tạo danh sách Log có khả năng tương tác (Clickable)
        logs.controls.clear()
        for idx, (desc, box) in enumerate(zip(details, boxes)):
            logs.controls.append(
                ft.GestureDetector(
                    # Khi click đúp -> Gọi hàm highlight_box
                    on_double_tap=lambda e, i=idx: highlight_box(i),
                    content=ft.Container(padding=5, ink=True, content=ft.Text(desc, size=12))
                )
            )
        preview.src_base64 = frame_to_base64(annotated)
        preview.update(); page.update()

    # [TÍNH NĂNG TƯƠNG TÁC] Highlight khuôn mặt
    def highlight_box(idx: int):
        if analyzed_image[0] is None or idx >= len(photo_boxes[0]): return
        
        img = analyzed_image[0].copy()
        x, y, w, h = photo_boxes[0][idx]
        
        # 1. Vẽ khung ĐỎ
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        if idx < len(face_labels[0]):
            # Lấy chuỗi full: "happy (0.95)"
            full_text = face_labels[0][idx]
            
            # [UI BÊN NGOÀI] Hiển thị full thông tin có điểm số
            label_text.value = full_text 
            
            # [TRÊN ẢNH] Chỉ lấy tên cảm xúc để vẽ
            # Tách chuỗi "happy (0.95)" thành ["happy", "(0.95)"] và lấy phần đầu
            emotion_only = full_text.split('(')[0].strip()
            
            # Vẽ chữ ĐỎ lên ảnh (chỉ tên cảm xúc)
            cv2.putText(
                img, 
                emotion_only, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 0, 255), 
                2
            )

        preview.src_base64 = frame_to_base64(img)
        preview.update()
        page.update()

    picker = ft.FilePicker(on_result=on_file_result)
    page.overlay.append(picker)

    picker = ft.FilePicker(on_result=on_file_result)
    page.overlay.append(picker)

    def save_to_storage(_):
        if analyzed_image[0] is None: return
        save_image_with_label(analyzed_image[0], label_text.value, "photo")
        page.snack_bar = ft.SnackBar(ft.Text("Đã lưu vào storage/")); page.snack_bar.open = True; page.update()

    return ft.View(
        route="/photo",
        controls=[
            ft.AppBar(title=ft.Text("Nhận diện 1 khung hình"), leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=back)),
            ft.Row([
                ft.Container(preview, expand=True, border=ft.border.all(1, ft.Colors.GREY)),
                ft.Column([
                    ft.Text("Cảm xúc"), label_text,
                    ft.ElevatedButton("Chọn ảnh", on_click=lambda _: picker.pick_files(allow_multiple=False)),
                    ft.ElevatedButton("Lưu vào thư viện", on_click=save_to_storage)
                ], expand=True, spacing=10)
            ], expand=True),
            ft.Text("Kéo thanh dưới để xem chi tiết (hoặc chạm thanh nhỏ để mở)"), ft.Container(height=4), ft.Row([peek_bar], alignment="center"),
        ],
        vertical_alignment="start",
    )


def storage_cards(selected: List[str], refresh_callback):
    """Tạo danh sách thẻ ảnh (Card) trong kho lưu trữ"""
    files = [f for f in os.listdir(STORAGE_DIR) if f.lower().endswith(".png")]
    cards = []
    for fname in sorted(files, reverse=True):
        path = os.path.join(STORAGE_DIR, fname)
        try:
            img = Image.open(path)
            thumbnail = img.copy()
            thumbnail.thumbnail((180, 140)) # Tạo ảnh nhỏ thumbnail để load nhanh
            buf = io.BytesIO()
            thumbnail.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            label = fname.split("_")[0]
            
            cards.append(ft.Card(content=ft.Container(
                padding=10, on_click=lambda e, p=path: (selected.clear(), selected.append(p), refresh_callback()),
                content=ft.Column([ft.Image(src_base64=b64, width=180, height=140, fit=ft.ImageFit.COVER), ft.Text(label, weight="bold"), ft.Text(fname, size=12)])
            )))
        except: pass
    return cards

def storage_view(page: ft.Page) -> ft.View:
    selected: List[str] = []
    grid = ft.GridView(expand=True, runs_count=3, max_extent=220, spacing=10, run_spacing=10)

    def refresh():
        grid.controls = storage_cards(selected, refresh)
        delete_btn.disabled = export_btn.disabled = not selected
        page.update()

    def back(_): page.go("/")
    def delete_file(_):
        if selected:
            try: os.remove(selected[0]); selected.clear(); refresh()
            except: pass
    
    def export_file(_):
        if selected:
            img = cv2.imread(selected[0])
            label = os.path.basename(selected[0]).split("_")[0]
            new_path = save_image_with_label(img, label, "export")
            page.snack_bar = ft.SnackBar(ft.Text(f"Đã xuất: {new_path}")); page.snack_bar.open = True; page.update(); refresh()

    delete_btn = ft.ElevatedButton("Xóa", icon=ft.Icons.DELETE, disabled=True, on_click=delete_file)
    export_btn = ft.ElevatedButton("Xuất", icon=ft.Icons.OUTBOX, disabled=True, on_click=export_file)
    refresh()

    return ft.View(route="/storage", controls=[ft.AppBar(title=ft.Text("Lưu trữ"), leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=back)), ft.Row([delete_btn, export_btn], spacing=10, alignment="start"), grid])


# -------------------------- 4. ĐIỂM BẮT ĐẦU (APP ENTRY) -------------------------- #

def main(page: ft.Page):
    """Hàm main: Thiết lập cửa sổ và điều hướng"""
    page.title = "Emotion Tracker"
    page.horizontal_alignment = "center"; page.vertical_alignment = "center"
    page.window_width = 1000; page.window_height = 720
    page.theme_mode = ft.ThemeMode.LIGHT; page.bgcolor = ft.Colors.WHITE    

    def route_change(e: ft.RouteChangeEvent):
        if e.route != "/live": live_state.running = False # Tắt camera khi rời trang Live
        page.overlay.clear(); page.views.clear()
        if page.route == "/": page.views.append(home_view(page))
        elif page.route == "/live": page.views.append(live_view(page))
        elif page.route == "/photo": page.views.append(photo_view(page))
        elif page.route == "/storage": page.views.append(storage_view(page))
        else: page.views.append(home_view(page))
        page.update()

    page.on_route_change = route_change
    page.on_view_pop = lambda _: page.go(page.views[-1].route if len(page.views) > 1 else "/")
    page.go(page.route or "/")

if __name__ == "__main__":
    ft.app(target=main)