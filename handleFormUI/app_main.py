import sys
import os
import re
import traceback
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSlot

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))  
if project_root not in sys.path: sys.path.insert(0, project_root)

from ui.ui_form_FaceRecognition import Ui_MainWindow 
try:
    from handleFormUI.worker import RecognitionWorker 
    from handleFormUI.add_user import AddUserDialog
except ImportError as e_fallback:
    QMessageBox.critical(None, "Lỗi Import", f"Không thể tải module: {e_fallback}.\nỨng dụng thoát."); sys.exit(1)

DATABASE_FOLDER_APP = os.path.join(project_root, 'database')
YOLO_MODEL_WEIGHTS_PATH = os.path.join(project_root, 'yolov5/runs/train/train_face_detection_v2/weights/best.pt')
YOLO_REPO_PATH = os.path.join(project_root, 'yolov5')
YOLO_CONFIDENCE_THRESHOLD = 0.4

os.makedirs(DATABASE_FOLDER_APP, exist_ok=True)

class FaceRecognitionApp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Nhận Diện Khuôn Mặt")
        self.recognition_thread = None
        self.add_user_window = None
        self.initialize_recognition_worker()
        self.btnAddPerson.clicked.connect(self.action_open_add_user_form)
        self.labelPicturePerson.setAlignment(Qt.AlignCenter)
        self.clear_recognized_person_info()

    def initialize_recognition_worker(self):
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.recognition_thread.stop(); self.recognition_thread.wait()
        
        self.recognition_thread = RecognitionWorker(
            yolo_weights_path=YOLO_MODEL_WEIGHTS_PATH,
            yolo_repo_path=YOLO_REPO_PATH,
            db_path=DATABASE_FOLDER_APP, # Đường dẫn database gốc được truyền vào worker
            yolo_confidence=YOLO_CONFIDENCE_THRESHOLD,
            parent=self
        )
        self.recognition_thread.signals.frame_ready.connect(self.update_camera_display)
        self.recognition_thread.signals.recognition_result.connect(self.display_recognition_result)
        self.recognition_thread.signals.no_recognition.connect(self.clear_recognized_person_info)
        self.recognition_thread.signals.error.connect(self.handle_worker_error)
        
        if hasattr(self.recognition_thread, '_prevent_run') and self.recognition_thread._prevent_run:
            QMessageBox.critical(self, "Lỗi Worker", "Worker không thể khởi tạo. Kiểm tra console.")
            self.btnAddPerson.setEnabled(False)
            self.labelCamera.setText("Lỗi Worker!")
        else:
            self.recognition_thread.start()
            self.statusBar().showMessage("Đang khởi động hệ thống...")

    @pyqtSlot(QImage)
    def update_camera_display(self, qt_image):
        if hasattr(self, 'labelCamera'):
            try:
                self.labelCamera.setPixmap(QPixmap.fromImage(qt_image).scaled(self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception as e: print(f"Lỗi update camera (main): {e}")

    @pyqtSlot(np.ndarray, str, str) 
    def display_recognition_result(self, face_crop_bgr, recognized_name, recognized_folder_name):
        self.txt_name_person.setText(recognized_name)
        match_id = re.match(r"^(\d+)_?(.*)", recognized_folder_name)
        id_str = match_id.group(1) if match_id else "N/A"
        self.txt_id_person.setText(id_str)
        photo_path = None
        person_folder = os.path.join(DATABASE_FOLDER_APP, recognized_folder_name)
        if os.path.isdir(person_folder):
            try:
                for file in os.listdir(person_folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        photo_path = os.path.join(person_folder, file); break
            except Exception as e: print(f"Lỗi tìm ảnh: {e}")
        if photo_path and os.path.exists(photo_path):
            pixmap = QPixmap(photo_path)
            if not pixmap.isNull(): self.labelPicturePerson.setPixmap(pixmap.scaled(self.labelPicturePerson.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else: self.labelPicturePerson.setText("Ảnh lỗi")
        else: self.labelPicturePerson.setText("Không có ảnh"); self.labelPicturePerson.setPixmap(QPixmap())

    @pyqtSlot()
    def clear_recognized_person_info(self):
        if self.txt_id_person.toPlainText() or self.txt_name_person.toPlainText():
            self.txt_name_person.setText("---"); self.txt_id_person.setText("---")
            self.labelPicturePerson.setText("(Chưa nhận diện)"); self.labelPicturePerson.setPixmap(QPixmap())

    @pyqtSlot(str)
    def handle_worker_error(self, error_message):
        self.statusBar().showMessage(f"Lỗi Worker: {error_message}", 7000)

    def action_open_add_user_form(self):
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.recognition_thread.stop(); self.labelCamera.setText("Camera tạm dừng...")
        
        # Truyền đường dẫn database gốc vào AddUserDialog
        self.add_user_window = AddUserDialog(database_main_folder_path=DATABASE_FOLDER_APP, parent=self) 
        self.add_user_window.user_added_completed.connect(self.handle_new_user_completed)
        self.add_user_window.request_return_to_main.connect(self.handle_return_from_add_user)
        self.hide(); self.add_user_window.show()

    @pyqtSlot()
    def handle_new_user_completed(self):
        if self.recognition_thread: self.recognition_thread.reload_embeddings()
        if self.add_user_window: self.add_user_window.close(); self.add_user_window = None
        self.show_main_window_and_restart_worker()

    @pyqtSlot()
    def handle_return_from_add_user(self):
        if self.add_user_window: self.add_user_window = None
        self.show_main_window_and_restart_worker()

    def show_main_window_and_restart_worker(self):
        self.show(); self.clear_recognized_person_info()
        if self.recognition_thread and not self.recognition_thread.isRunning():
            if hasattr(self.recognition_thread, '_prevent_run') and self.recognition_thread._prevent_run:
                 QMessageBox.warning(self, "Lỗi Worker", "Worker lỗi. Kiểm tra console.")
                 self.labelCamera.setText("Lỗi Worker!")
            else:
                self.recognition_thread.start()
                self.statusBar().showMessage("Đang khởi động lại nhận diện...")
        elif not self.recognition_thread:
            self.initialize_recognition_worker()

    def closeEvent(self, event):
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.recognition_thread.stop(); self.recognition_thread.wait()
        if self.add_user_window: self.add_user_window.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app_window = FaceRecognitionApp()
    main_app_window.show()
    sys.exit(app.exec_())