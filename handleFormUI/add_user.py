import cv2
import sys
import os
import numpy as np
import traceback
import re
from PyQt5.QtWidgets import QDialog, QMessageBox, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from ui.ui_form_ChupAnh import Ui_Form

MAX_IMAGES_PER_USER = 8

class AddUserDialog(QDialog, Ui_Form):
    user_added_completed = pyqtSignal()
    request_return_to_main = pyqtSignal()

    def __init__(self, database_main_folder_path: str, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Thêm Người Dùng Mới")
        self.db_path = database_main_folder_path
        self.capture = None
        self.timer_camera_preview = QTimer(self)
        self.timer_camera_preview.timeout.connect(self.update_camera_preview_feed)
        self.captured_frame_for_review = None
        self.temp_images_list = []
        self.current_image_count = 0
        self.btnChupAnh.clicked.connect(self.action_capture_image)
        self.btnDongY.clicked.connect(self.action_accept_image)
        self.btnChupLai.clicked.connect(self.action_recapture_image)
        self.btnHoanTat.clicked.connect(self.action_complete_registration)
        self.btnReset.clicked.connect(self.action_reset_form)
        self.btnQuayLai.clicked.connect(self.action_go_back)
        self.txtTenNguoiMoi.textChanged.connect(self.check_name_and_activate_capture)
        self.initialize_camera()
        self.reset_form_to_initial_state()
        if not os.path.exists(self.db_path):
            try:
                os.makedirs(self.db_path)
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể tạo thư mục '{self.db_path}': {e}")

    def initialize_camera(self):
        try:
            for cam_id in [0, 1, -1]:
                self.capture = cv2.VideoCapture(cam_id)
                if self.capture and self.capture.isOpened(): break
            if not self.capture or not self.capture.isOpened(): raise ValueError("Không thể mở camera.")
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.timer_camera_preview.start(30)
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể mở webcam: {e}"); self.capture = None

    def update_camera_preview_feed(self):
        if self.capture and self.capture.isOpened() and self.timer_camera_preview.isActive():
            ret, frame_bgr = self.capture.read()
            if ret:
                try:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape; bytes_per_line = ch * w
                    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.labelCamera.setPixmap(QPixmap.fromImage(qt_image).scaled(self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                except Exception as e: print(f"Lỗi update preview: {e}")

    def display_captured_image(self, frame_bgr):
        if frame_bgr is None: return
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape; bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.labelCamera.setPixmap(QPixmap.fromImage(qt_image).scaled(self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e: QMessageBox.warning(self, "Lỗi", f"Lỗi hiển thị ảnh chụp: {e}")

    def reset_form_to_initial_state(self):
        self.temp_images_list.clear(); self.current_image_count = 0
        self.captured_frame_for_review = None
        self.txtTenNguoiMoi.clear(); self.txtTenNguoiMoi.setEnabled(True)
        self.labelTookPhoto.setText(f"Ảnh đã chấp nhận: {self.current_image_count}/{MAX_IMAGES_PER_USER}")
        self.btnChupAnh.show(); self.btnChupAnh.setEnabled(False)
        self.btnDongY.hide(); self.btnChupLai.hide(); self.btnHoanTat.hide()
        self.btnReset.show(); self.btnQuayLai.show()
        if self.capture and self.capture.isOpened() and not self.timer_camera_preview.isActive():
            self.timer_camera_preview.start(30)
        self.labelCamera.clear()
        if not (self.capture and self.capture.isOpened()): self.labelCamera.setText("Lỗi Camera!")

    def check_name_and_activate_capture(self):
        user_name = self.txtTenNguoiMoi.text().strip()
        can_capture = bool(user_name) and (self.capture is not None and self.capture.isOpened())
        if self.btnChupAnh.isVisible(): self.btnChupAnh.setEnabled(can_capture)

    def action_capture_image(self):
        if not (self.capture and self.capture.isOpened()): QMessageBox.warning(self, "Lỗi", "Camera chưa sẵn sàng."); return
        if not self.txtTenNguoiMoi.text().strip(): QMessageBox.warning(self, "Lỗi", "Nhập tên người dùng."); return
        ret, frame_bgr = self.capture.read()
        if ret and frame_bgr is not None:
            self.timer_camera_preview.stop()
            self.captured_frame_for_review = frame_bgr.copy()
            self.display_captured_image(self.captured_frame_for_review)
            self.btnChupAnh.hide(); self.btnDongY.show(); self.btnDongY.setEnabled(True)
            self.btnChupLai.show(); self.btnChupLai.setEnabled(True)
            self.txtTenNguoiMoi.setEnabled(False)
        else: QMessageBox.warning(self, "Lỗi", "Không thể chụp ảnh.")

    def action_accept_image(self):
        if self.captured_frame_for_review is None: return
        self.temp_images_list.append(self.captured_frame_for_review)
        self.current_image_count = len(self.temp_images_list)
        self.labelTookPhoto.setText(f"Ảnh đã chấp nhận: {self.current_image_count}/{MAX_IMAGES_PER_USER}")
        self.captured_frame_for_review = None
        if self.current_image_count < MAX_IMAGES_PER_USER:
            self.timer_camera_preview.start(30); self.btnChupAnh.show(); self.btnChupAnh.setEnabled(True)
            self.btnDongY.hide(); self.btnChupLai.hide()
        else:
            self.timer_camera_preview.stop()
            self.labelCamera.setText(f"Đã đủ {MAX_IMAGES_PER_USER} ảnh.\nVui lòng nhấn hoàn tất để lưu.") # Cập nhật thông báo
            self.btnChupAnh.hide(); self.btnDongY.hide(); self.btnChupLai.hide()
            self.btnHoanTat.show(); self.btnHoanTat.setEnabled(True)

    def action_recapture_image(self):
        self.captured_frame_for_review = None; self.timer_camera_preview.start(30)
        self.btnChupAnh.show(); self.btnChupAnh.setEnabled(True)
        self.btnDongY.hide(); self.btnChupLai.hide()

    def find_next_available_id(self):
        if not os.path.exists(self.db_path) or not os.path.isdir(self.db_path): return 1
        existing_ids = set()
        try:
            for item_name in os.listdir(self.db_path):
                item_path = os.path.join(self.db_path, item_name)
                if os.path.isdir(item_path):
                    match = re.match(r"^(\d+)_.*", item_name)
                    if match:
                        try: existing_ids.add(int(match.group(1)))
                        except ValueError: pass
        except Exception as e: print(f"Lỗi quét ID: {e}"); return None
        current_id = 1
        while current_id in existing_ids: current_id += 1
        return current_id

    def action_complete_registration(self):
        user_name_raw = self.txtTenNguoiMoi.text().strip()
        user_name_processed = '_'.join(''.join(c for c in user_name_raw if c.isalnum() or c == ' ').strip().split())
        if not user_name_processed: QMessageBox.warning(self, "Lỗi", "Tên không hợp lệ."); return
        if len(self.temp_images_list) != MAX_IMAGES_PER_USER: QMessageBox.warning(self, "Lỗi", f"Cần {MAX_IMAGES_PER_USER} ảnh."); return

        next_id = self.find_next_available_id()
        if next_id is None: QMessageBox.critical(self, "Lỗi", "Không thể tạo ID."); return
        
        folder_id_str = f"{next_id:02d}"
        folder_name_final = f"{folder_id_str}_{user_name_processed}"
        user_folder_path = os.path.join(self.db_path, folder_name_final)

        if os.path.exists(user_folder_path): QMessageBox.warning(self, "Lỗi", f"Thư mục '{folder_name_final}' đã tồn tại."); return
            
        try:
            os.makedirs(user_folder_path, exist_ok=True)
            for i, img_data in enumerate(self.temp_images_list):
                image_filename = f"{user_name_processed}_{i+1:02d}.jpg"
                save_path = os.path.join(user_folder_path, image_filename)
                if not cv2.imwrite(save_path, img_data): raise IOError(f"Lưu ảnh {i+1} thất bại.")
                print(f"Đã lưu: {save_path}")
            
            QMessageBox.information(self, "Thành Công", f"Đã thêm '{user_name_raw}' (ID: {folder_id_str}).")
            self.user_added_completed.emit()
            self.action_go_back()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi lưu trữ: {e}"); traceback.print_exc()
    
    def action_reset_form(self):
        if QMessageBox.question(self, 'Reset', "Xóa ảnh đã chụp và bắt đầu lại?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            self.reset_form_to_initial_state()

    def action_go_back(self):
        if self.timer_camera_preview.isActive(): self.timer_camera_preview.stop()
        if self.capture: self.capture.release(); self.capture = None
        self.temp_images_list.clear()
        self.request_return_to_main.emit()
        self.reject() 

    def closeEvent(self, event):
        if self.timer_camera_preview.isActive(): self.timer_camera_preview.stop()
        if self.capture: self.capture.release(); self.capture = None
        super().closeEvent(event)