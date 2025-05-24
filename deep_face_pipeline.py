# deep_face_pipeline.py
import cv2
from deepface import DeepFace
from yolo_face_detector import YOLOFaceDetector # Đảm bảo file này tồn tại và đúng
import os
import time
import numpy as np

class FaceRecognitionPipeline:
    # ... (Phần __init__ và các hàm khác giữ nguyên như bạn đã cung cấp) ...
    def __init__(self, yolo_model_weights_path, db_path,
                 yolo_repo_path='yolov5',
                 df_model_name='Facenet',
                 df_distance_metric='cosine',
                 yolo_confidence=0.5,
                 use_alignment=True,
                 alignment_detector_backend='retinaface'):

        self.yolo_detector = YOLOFaceDetector(
            yolo_repo_path=yolo_repo_path,
            model_weights_path=yolo_model_weights_path,
            confidence_threshold=yolo_confidence
        )
        self.db_path = db_path
        self.df_model_name = df_model_name
        self.df_distance_metric = df_distance_metric
        self.use_alignment = use_alignment
        self.alignment_detector_backend = alignment_detector_backend

        if not os.path.exists(self.db_path):
            print(f"Warning: Database path '{self.db_path}' does not exist. This may cause issues.")
        else:
            print(f"DeepFace database path set to: {self.db_path}")

        print(f"DeepFace model '{self.df_model_name}' with metric '{self.df_distance_metric}' will be used.")
        if self.use_alignment:
            print(f"Face alignment enabled using '{self.alignment_detector_backend}' backend.")
        else:
            print("Face alignment disabled.")


    def _get_aligned_face(self, face_image_bgr):
        try:
            extracted_data = DeepFace.extract_faces(
                img_path=face_image_bgr,
                detector_backend=self.alignment_detector_backend,
                enforce_detection=True,
                align=True
            )
            if extracted_data and len(extracted_data) > 0 and extracted_data[0]['confidence'] > 0.5:
                aligned_face_rgb_float = extracted_data[0]['face']
                aligned_face_bgr_uint8 = (aligned_face_rgb_float * 255).astype(np.uint8)
                aligned_face_bgr_uint8 = cv2.cvtColor(aligned_face_bgr_uint8, cv2.COLOR_RGB2BGR)
                return aligned_face_bgr_uint8
            else:
                return None
        except Exception as e:
            if self.alignment_detector_backend != 'opencv':
                try:
                    extracted_data_fallback = DeepFace.extract_faces(
                        img_path=face_image_bgr,
                        detector_backend='opencv',
                        enforce_detection=True,
                        align=True
                    )
                    if extracted_data_fallback and len(extracted_data_fallback) > 0 and extracted_data_fallback[0]['confidence'] > 0:
                        aligned_face_rgb_float_fb = extracted_data_fallback[0]['face']
                        aligned_face_bgr_uint8_fb = (aligned_face_rgb_float_fb * 255).astype(np.uint8)
                        aligned_face_bgr_uint8_fb = cv2.cvtColor(aligned_face_bgr_uint8_fb, cv2.COLOR_RGB2BGR)
                        return aligned_face_bgr_uint8_fb
                except Exception as e_fb:
                    pass
            return None


    def process_image(self, image_path_or_array, output_image_path=None):
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                print(f"Error: Could not read image from {image_path_or_array}")
                return None, []
        else:
            image = image_path_or_array.copy()

        if self.yolo_detector.model is None:
            print("YOLO model not loaded. Cannot process image.")
            return image, []

        bboxes, cropped_faces = self.yolo_detector.detect_faces(image)
        recognition_results = []
        image_to_draw_on = image.copy()

        for i, (bbox, yolo_cropped_face) in enumerate(zip(bboxes, cropped_faces)):
            face_to_recognize = yolo_cropped_face
            if yolo_cropped_face is None or yolo_cropped_face.size == 0:
                recognition_results.append({"bbox": bbox, "name": "ErrorCrop", "distance": -1})
                label = "ErrorCrop"
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image_to_draw_on, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue

            if self.use_alignment:
                aligned_face = self._get_aligned_face(yolo_cropped_face)
                if aligned_face is not None and aligned_face.size > 0:
                    face_to_recognize = aligned_face
            try:
                dfs = DeepFace.find(img_path=face_to_recognize,
                                    db_path=self.db_path,
                                    model_name=self.df_model_name,
                                    distance_metric=self.df_distance_metric,
                                    enforce_detection=False,
                                    silent=True)
                identity_df = dfs[0] if isinstance(dfs, list) and len(dfs) > 0 and not dfs[0].empty else None
                name = "Unknown"
                distance_val = -1.0
                if identity_df is not None:
                    best_match = identity_df.iloc[0]
                    identity_path = best_match['identity']
                    name = os.path.basename(os.path.dirname(identity_path))
                    distance_col_candidate1 = 'distance'
                    distance_col_candidate2 = f'{self.df_model_name}_{self.df_distance_metric}'
                    if distance_col_candidate1 in best_match.index:
                        distance_val = best_match[distance_col_candidate1]
                    elif distance_col_candidate2 in best_match.index:
                        distance_val = best_match[distance_col_candidate2]
                    else:
                        name = "ErrorDistCol"
                    threshold = 0.40
                    if self.df_model_name == 'VGG-Face': threshold = 0.40
                    elif self.df_model_name == 'Facenet512': threshold = 0.55
                    elif self.df_model_name == 'ArcFace': threshold = 0.68
                    if distance_val > threshold or name == "ErrorDistCol":
                        name = "Unknown" if name != "ErrorDistCol" else name
                recognition_results.append({"bbox": bbox, "name": name, "distance": distance_val})
                label = f"{name} ({distance_val:.2f})" if distance_val != -1.0 else name
            except Exception as e:
                recognition_results.append({"bbox": bbox, "name": "ErrorRec", "distance": -1})
                label = "ErrorRec"
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if "Error" not in label and label != "Unknown" else ((0,165,255) if label == "Unknown" else (0,0,255))
            cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_to_draw_on, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if output_image_path:
            cv2.imwrite(output_image_path, image_to_draw_on)
        return image_to_draw_on, recognition_results


    # THAY ĐỔI TRONG HÀM NÀY
    def process_video(self, video_source, output_video_path=None, 
                      display_width=640, display_height=480,
                      camera_capture_width=640, camera_capture_height=480): # Thêm tham số cho độ phân giải capture
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Trying alternative backend (CAP_DSHOW) for video source {video_source}...")
            cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW) # Thử backend DSHOW

        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source} even with DSHOW backend.")
            return

        # CỐ GẮNG ĐẶT ĐỘ PHÂN GIẢI CAPTURE CỦA CAMERA
        # Chỉ thực hiện nếu video_source là một số nguyên (ID camera)
        if isinstance(video_source, int):
            print(f"Attempting to set camera capture resolution to: {camera_capture_width}x{camera_capture_height}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_capture_height)
            
            # Đọc lại để kiểm tra độ phân giải thực tế
            actual_capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Actual camera capture resolution set to: {actual_capture_width}x{actual_capture_height}")
        else:
            # Nếu là file video, lấy độ phân giải từ file
            actual_capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Processing video file with resolution: {actual_capture_width}x{actual_capture_height}")


        writer = None
        if output_video_path:
            # Sử dụng độ phân giải thực tế của camera/video để ghi file
            fps_in = cap.get(cv2.CAP_PROP_FPS)
            # Đặt FPS mặc định nếu không lấy được hoặc quá thấp/cao
            if not (0 < fps_in <= 120): fps_in = 30.0 
            fps_out = fps_in # Ghi với FPS gốc hoặc FPS đã điều chỉnh
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps_out, (actual_capture_width, actual_capture_height))
            print(f"Output video: {output_video_path} at {fps_out:.2f} FPS, resolution: {actual_capture_width}x{actual_capture_height}")

        processing_times = []

        while True:
            ret, frame = cap.read() 
            if not ret:
                print("End of video stream or cannot read frame.")
                break

            # Nếu frame đọc được có kích thước không mong muốn (do cap.set không thành công hoàn toàn)
            # và bạn MUỐN ép nó về kích thước xử lý cố định (ví dụ 640x480) trước khi đưa vào YOLO
            # thì có thể resize frame ở đây. Tuy nhiên, điều này có thể làm mất thông tin nếu frame gốc lớn hơn.
            # Thường thì nên để YOLO xử lý frame ở kích thước nó nhận được từ camera (actual_capture_width, actual_capture_height)
            # Ví dụ: nếu muốn ép kích thước xử lý:
            # if frame.shape[1] != camera_capture_width or frame.shape[0] != camera_capture_height:
            #     frame = cv2.resize(frame, (camera_capture_width, camera_capture_height), interpolation=cv2.INTER_AREA)

            frame_process_start_time = time.time()
            processed_frame_original_size, _ = self.process_image(frame)
            processing_times.append(time.time() - frame_process_start_time)

            if processed_frame_original_size is not None:
                if processing_times:
                    if len(processing_times) > 30: 
                        processing_times.pop(0)
                    avg_processing_time = sum(processing_times) / len(processing_times)
                    processing_fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
                    cv2.putText(processed_frame_original_size, f"Avg Proc. FPS: {processing_fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            frame_to_display = processed_frame_original_size
            if processed_frame_original_size is not None and \
               processed_frame_original_size.shape[0] > 0 and processed_frame_original_size.shape[1] > 0:
                h_orig, w_orig = processed_frame_original_size.shape[:2]
                ratio = min(display_width / w_orig, display_height / h_orig) if w_orig > 0 and h_orig > 0 else 1.0
                new_w = int(w_orig * ratio)
                new_h = int(h_orig * ratio)
                if new_w > 0 and new_h > 0 :
                    frame_to_display = cv2.resize(processed_frame_original_size, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if frame_to_display is not None:
                 cv2.imshow("Face Recognition", frame_to_display)
            
            if writer and processed_frame_original_size is not None:
                writer.write(processed_frame_original_size)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Video processing finished.")