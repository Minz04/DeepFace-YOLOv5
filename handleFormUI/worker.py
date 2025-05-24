import cv2
import numpy as np
import time
import os
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage
import traceback
import re 

from deepface import DeepFace 
try:
    # Sử dụng YOLOFaceDetector từ project của bạn
    from yolo_face_detector import YOLOFaceDetector
    YOLO_AVAILABLE = True
except ImportError:
    print("[LỖI WORKER] Không thể nhập YOLOFaceDetector.")
    YOLO_AVAILABLE = False
    class YOLOFaceDetector:
        def __init__(self, *args, **kwargs): self.model = None
        def detect_faces(self, *args, **kwargs): return [], []

# Cấu hình DeepFace (CẦN ĐỒNG BỘ VỚI main.py CŨ CỦA BẠN NẾU CÓ)
DEEPFACE_MODEL_NAME = 'Facenet'     # Model DeepFace bạn dùng
DEEPFACE_DETECTOR_BACKEND = 'opencv' # Backend phát hiện khuôn mặt cho DeepFace.find (nếu enforce_detection=True)
                                     # hoặc cho DeepFace.represent nếu cần
DEEPFACE_ALIGNMENT = True          # Có sử dụng alignment không
DEEPFACE_DISTANCE_METRIC = 'cosine' # Hoặc 'euclidean_l2'
DEEPFACE_RECOGNITION_THRESHOLD = 0.45 # NGƯỠNG QUAN TRỌNG! (cho cosine, < threshold là khớp)
                                     # Nếu dùng 'euclidean_l2', ngưỡng có thể là 1.0 - 1.2

DATABASE_FOLDER_WORKER = "database" # Sẽ được cập nhật

class RecognitionSignals(QObject):
    frame_ready = pyqtSignal(QImage)
    recognition_result = pyqtSignal(np.ndarray, str, str) # crop, name, folder_name
    no_recognition = pyqtSignal()
    error = pyqtSignal(str)

class RecognitionWorker(QThread):
    def __init__(self, yolo_weights_path: str, yolo_repo_path: str, 
                 db_path: str, yolo_confidence: float = 0.5, parent=None):
        super().__init__(parent)
        self.signals = RecognitionSignals()
        self.running = False
        self._prevent_run = False
        
        global DATABASE_FOLDER_WORKER
        DATABASE_FOLDER_WORKER = db_path

        if not YOLO_AVAILABLE:
            self._prevent_run = True; return

        try:
            self.yolo_detector = YOLOFaceDetector(
                yolo_repo_path=yolo_repo_path,
                model_weights_path=yolo_weights_path,
                confidence_threshold=yolo_confidence
            )
            if self.yolo_detector.model is None:
                raise ValueError("Model YOLO không tải được trong worker.")
            print("[WORKER] YOLOFaceDetector đã khởi tạo.")
            # Khởi tạo model DeepFace một lần để tải trước
            print(f"[WORKER] Đang khởi tạo model DeepFace '{DEEPFACE_MODEL_NAME}'...")
            # DeepFace.build_model(DEEPFACE_MODEL_NAME) # Tải model chính
            # Có thể gọi find với ảnh dummy để tải hết các thành phần
            dummy_img = np.zeros((100,100,3), dtype=np.uint8)
            try:
                DeepFace.represent(dummy_img, model_name=DEEPFACE_MODEL_NAME, enforce_detection=False)
            except: # Bắt lỗi nếu dummy image không hợp lệ với backend nào đó
                pass
            print(f"[WORKER] Model DeepFace '{DEEPFACE_MODEL_NAME}' có vẻ đã sẵn sàng.")

        except Exception as e:
            print(f"[LỖI WORKER] Khởi tạo YOLO/DeepFace: {e}"); self._prevent_run = True

    # Phương thức này sẽ được gọi từ GUI để yêu cầu worker reload embeddings
    def reload_embeddings(self):
        print("[WORKER] Yêu cầu reload embeddings...")
        deleted_any = False
        try:
            for item in os.listdir(DATABASE_FOLDER_WORKER):
                if item.endswith(".pkl"): # Xóa TẤT CẢ các file .pkl trong thư mục database
                    pkl_path = os.path.join(DATABASE_FOLDER_WORKER, item)
                    try:
                        os.remove(pkl_path)
                        print(f"[WORKER] Đã xóa file representation: {pkl_path}")
                        deleted_any = True
                    except Exception as e_del:
                        print(f"[WORKER] Lỗi khi xóa {pkl_path}: {e_del}")
            if not deleted_any:
                print("[WORKER] Không tìm thấy file .pkl nào để xóa trong reload_embeddings.")
        except Exception as e:
            print(f"[WORKER] Lỗi khi duyệt thư mục database để xóa .pkl: {e}")


    def run(self):
        if self._prevent_run:
            self.signals.error.emit("Worker không thể chạy do lỗi khởi tạo."); return

        self.running = True; cap = None
        try:
            for cam_id in [0, 1, -1]:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened(): break
            if not cap or not cap.isOpened(): raise IOError("Không thể mở camera.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            self.signals.error.emit(f"Lỗi camera: {e}"); self.running = False;
            if cap: cap.release(); return

        last_sent_identity_path = None
        last_recognition_time = 0

        while self.running:
            try:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None: time.sleep(0.05); continue

                frame_for_display = frame_bgr.copy()
                bboxes, cropped_faces_bgr = self.yolo_detector.detect_faces(frame_bgr)
                
                best_match_this_frame = None # (crop, name, folder, distance)
                min_dist_this_frame = float('inf')

                for i, (bbox, face_crop) in enumerate(zip(bboxes, cropped_faces_bgr)):
                    if face_crop is None or face_crop.size == 0: continue
                    try:
                        # DeepFace.find nhận ảnh numpy BGR
                        dfs_results = DeepFace.find(
                            img_path=face_crop, 
                            db_path=DATABASE_FOLDER_WORKER,
                            model_name=DEEPFACE_MODEL_NAME,
                            distance_metric=DEEPFACE_DISTANCE_METRIC,
                            detector_backend=DEEPFACE_DETECTOR_BACKEND, # Backend cho find nếu cần detect lại
                            align=DEEPFACE_ALIGNMENT,
                            enforce_detection=False, # Đã có face_crop
                            silent=True
                        )
                        
                        if dfs_results and not dfs_results[0].empty:
                            df_person = dfs_results[0].iloc[0] # Lấy người khớp nhất
                            
                            # Lấy cột distance, có thể là 'distance' hoặc 'Model_metric'
                            dist_col = 'distance'
                            if f"{DEEPFACE_MODEL_NAME}_{DEEPFACE_DISTANCE_METRIC}" in df_person.index:
                                dist_col = f"{DEEPFACE_MODEL_NAME}_{DEEPFACE_DISTANCE_METRIC}"
                            
                            current_distance = df_person[dist_col]

                            if current_distance < DEEPFACE_RECOGNITION_THRESHOLD:
                                identity_path_df = df_person['identity']
                                person_folder_name_df = os.path.basename(os.path.dirname(identity_path_df))
                                
                                match_id_name = re.match(r"^(\d+)_?(.*)", person_folder_name_df)
                                display_name_df = person_folder_name_df
                                if match_id_name:
                                    raw_name = match_id_name.group(2) if match_id_name.group(2) else "N/A"
                                    display_name_df = raw_name.replace('_', ' ')

                                if current_distance < min_dist_this_frame:
                                    min_dist_this_frame = current_distance
                                    best_match_this_frame = (face_crop.copy(), display_name_df, person_folder_name_df, identity_path_df)
                                
                                cv2.rectangle(frame_for_display, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
                                label = f"{display_name_df} ({current_distance:.2f})"
                                cv2.putText(frame_for_display, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                            else: # Unknown
                                cv2.rectangle(frame_for_display, tuple(bbox[:2]), tuple(bbox[2:]), (0,0,255), 2)
                                cv2.putText(frame_for_display, "Unknown", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
                        else: # Không có ai trong DB khớp
                            cv2.rectangle(frame_for_display, tuple(bbox[:2]), tuple(bbox[2:]), (0,0,255), 2)
                            cv2.putText(frame_for_display, "Unknown", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
                    except Exception as e_find:
                        print(f"Lỗi DeepFace.find: {e_find}"); #traceback.print_exc()
                        cv2.rectangle(frame_for_display, tuple(bbox[:2]), tuple(bbox[2:]), (255,0,0), 2)
                        cv2.putText(frame_for_display, "ErrorRec", (bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1)
                
                current_time = time.time()
                if best_match_this_frame:
                    crop_sig, name_sig, folder_sig, id_path_sig = best_match_this_frame
                    if id_path_sig != last_sent_identity_path or (current_time - last_recognition_time) > 2.0:
                        self.signals.recognition_result.emit(crop_sig, name_sig, folder_sig)
                        last_sent_identity_path = id_path_sig
                        last_recognition_time = current_time
                elif last_sent_identity_path is not None:
                    self.signals.no_recognition.emit()
                    last_sent_identity_path = None
                    last_recognition_time = current_time

                display_rgb = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)
                h, w, ch = display_rgb.shape
                q_img = QImage(display_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.signals.frame_ready.emit(q_img.copy())
                self.msleep(10)
            except Exception as e_loop:
                print(f"Lỗi chung worker: {e_loop}"); traceback.print_exc()
                self.signals.error.emit(f"Lỗi: {e_loop}"); self.msleep(500)
        if cap: cap.release(); print("[WORKER] Luồng đã dừng, camera released.")

    def stop(self):
        self.running = False
        print("[WORKER] Yêu cầu dừng luồng...")