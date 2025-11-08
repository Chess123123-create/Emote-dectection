# ===============================================
# main.py
# -----------------------------------------------
# File chính để khởi chạy ứng dụng Flet.
# ===============================================

import flet as ft           # Thư viện Flet – dùng để tạo giao diện người dùng.
from ui import AppUI        # Import lớp AppUI – phần giao diện chính của ứng dụng.

# -------------------------------------------------
# Hàm main: điểm bắt đầu của ứng dụng Flet.
# -------------------------------------------------
def main(page: ft.Page):
    app = AppUI(page)                  # Tạo đối tượng giao diện (AppUI) và gắn vào trang Flet.
    page.on_close = lambda e: app.clean_up()  # Khi người dùng đóng app, gọi hàm dọn dẹp (giải phóng camera, v.v.).
    page.update()                      # Cập nhật lại giao diện (render nội dung mới).

# -------------------------------------------------
# Cấu hình chạy ứng dụng Flet.
# -------------------------------------------------
if __name__ == "__main__":
    ft.app(target=main, view=ft.FLET_APP)  # Chạy app với hàm main, hiển thị trong cửa sổ Flet.
