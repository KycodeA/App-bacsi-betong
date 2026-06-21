import os
import cv2
import math
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Nạp mô hình ONNX mới mà bạn vừa huấn luyện và xuất ra từ Colab
print("🚀 Đang khởi tạo mô hình YOLOv8 BTXM Enhanced...")
model = YOLO('yolov8n.onnx', task='detect') 

def birds_eye_view(frame):
    """Bẻ phối cảnh mặt đường để đưa về ma trận nhìn từ trên xuống (Hệ mét)"""
    height, width = frame.shape[:2]
    # 4 điểm ghim phối cảnh tùy chỉnh theo góc quay camera thực tế
    pts1 = np.float32([[width * 0.25, height * 0.55], [width * 0.75, height * 0.55], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, matrix, (width, height))

def process_video_segments(filepath, lane_width=3.5, sub_segments_count=10):
    """
    Xử lý video dài 500m, cắt frame, nhận diện nứt và gom cụm thành 10 phân đoạn 50m.
    Trả về danh sách diện tích nứt và tỷ lệ % của từng phân đoạn.
    """
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return [0.0] * sub_segments_count
        
    # Tính toán số lượng frame cho mỗi phân đoạn 50m (Chia đều video làm 10 phần)
    frames_per_segment = math.ceil(total_frames / sub_segments_count)
    
    # Khởi tạo mảng lưu tổng diện tích vết nứt cho 10 phân đoạn (mỗi phân đoạn 50m)
    segment_crack_areas = [0.0] * sub_segments_count
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Tối ưu tốc độ: Cứ 15 frames (khoảng 0.5 giây) ta mới trích xuất 1 ảnh để phân tích AI
        if frame_idx % 15 == 0:
            # Xác định frame này đang thuộc phân đoạn 50m thứ mấy (0 đến 9)
            current_seg_idx = min(frame_idx // frames_per_segment, sub_segments_count - 1)
            
            # Tiền xử lý ảnh: Resize và Bẻ phối cảnh phẳng (BEV)
            frame_resized = cv2.resize(frame, (320, 320))
            warped_frame = birds_eye_view(frame_resized)
            
            # Chạy AI dự đoán vết nứt
            results = model(warped_frame, imgsz=320, verbose=False)
            
            frame_max_area = 0.0 # Diện tích nứt lớn nhất ghi nhận được trong frame này
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id, 'D00') # Mặc định lấy theo nhãn RDD2022
                
                # Quy đổi kích thước bounding box từ pixel sang mét thực tế
                # Giả định khung nhìn BEV bao phủ chiều rộng làn đường (lane_width) và chiều dài 5.0m
                w_m = (box.xywh[0][2].item() / 320) * lane_width
                h_m = (box.xywh[0][3].item() / 320) * 5.0
                
                # 🛠️ BỘ LỌC LOẠI BỎ KHE NỐI ĐƯỜNG BTXM (Nguyên tắc đề bài)
                if label == 'D10': # Nứt ngang
                    # Nếu vết nứt quá dài và mảnh nằm ngang tuyệt đối -> Bỏ qua vì là khe co giãn
                    if w_m > 2.8 and h_m < 0.15: 
                        continue
                if label == 'D00': # Nứt dọc
                    x_center_ratio = box.xywhn[0][0].item()
                    # Nếu vết nứt bám sát rìa mép dọc đường -> Bỏ qua vì là khe nối dọc tấm
                    if x_center_ratio < 0.06 or x_center_ratio > 0.94: 
                        continue
                
                # 📐 TÍNH TOÁN DIỆN TÍCH THEO QUY ĐỊNH
                if label == 'D20': # Nứt hình lưới (Alligator Crack)
                    frame_max_area += (w_m * h_m)
                elif label in ['D00', 'D10']: # Các vết nứt đơn (ngang, dọc, chéo)
                    length_m = math.sqrt(w_m**2 + h_m**2)
                    frame_max_area += (length_m * 0.3) # Diện tích = chiều dài x 0.3m
            
            # Cộng dồn diện tích hỏng hóc vào phân đoạn tương ứng
            segment_crack_areas[current_seg_idx] += frame_max_area
            
        frame_idx += 1
        
    cap.release()
    
    # Tính toán Tỷ lệ % Diện tích nứt dựa trên diện tích phân đoạn (50m x Chiều rộng làn)
    # S_phân_đoạn = 50 * 3.5 = 175 m2
    segment_area_fixed = 50.0 * lane_width
    segment_percentages = []
    
    for crack_area in segment_crack_areas:
        # Tỷ lệ % = (Diện tích nứt / Diện tích phân đoạn) * 100
        pct = (crack_area / segment_area_fixed) * 100
        # Ràng buộc kết quả tối đa là 100%
        segment_percentages.append(round(min(pct, 100.0), 2))
        
    return segment_percentages

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    try:
        lane_width = float(request.form.get('lane_width', '3.5') or 3.5)
        file = request.files.get('video_file')
        
        if not file or file.filename == '':
            return jsonify({'status': 'error', 'message': 'Vui lòng chọn file video'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Chạy phân tích ra mảng 10 số liệu tương ứng cho 10 phân đoạn 50m
        percentages = process_video_segments(filepath, lane_width=lane_width)
        
        # Xóa file tạm sau khi phân tích xong để nhẹ máy
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            'status': 'success',
            'filename': filename,
            'percentages': percentages # Trả về mảng 10 số liệu ví dụ: [0.5, 0.0, 1.2, ..., 4.5]
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Chạy ứng dụng nội bộ tại port 7860 hoặc 5000
    app.run(host='0.0.0.0', port=7860, debug=True)
