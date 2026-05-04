import numpy as np
from PIL import Image

def process_apple_prediction(image, model, conf_threshold=0.4):
    """
    Khai thác sức mạnh của Ultralytics engine để dự đoán, 
    vẽ mask segmentation và thống kê chi tiết.
    """
    # 1. Chinh phạt dữ liệu: model.predict tự động lo toàn bộ Pre-processing
    # Ép kích thước imgsz=640 để đồng nhất với lúc train
    results = model.predict(source=image, imgsz=640, conf=conf_threshold)
    result = results[0] # Lấy kết quả của bức ảnh đầu tiên (vì ta truyền vào 1 ảnh)

    # 2. Kết xuất hình ảnh: Dùng hàm .plot() tích hợp sẵn để vẽ Box và Mask
    # plot() trả về mảng numpy hệ màu BGR (chuẩn của OpenCV)
    annotated_img_bgr = result.plot()
    
    # Đảo ngược kênh màu từ BGR sang RGB để hiển thị đúng trên chuẩn của PIL/Gradio
    annotated_img_rgb = annotated_img_bgr[..., ::-1] 
    final_image = Image.fromarray(annotated_img_rgb)

    # 3. Thống kê cục diện: Phân tích các object đã nhận diện được
    names_dict = result.names # Ví dụ: {0: 'Green_Apple', 1: 'Red_Apple'}
    
    # Nếu không phát hiện được quả táo nào
    if result.boxes is None or len(result.boxes) == 0:
        return final_image, "⚠️ Không phát hiện quả táo nào trong khung hình. Thử hạ ngưỡng tin cậy (Confidence) xuống."

    # Lấy mảng chứa ID của các class nhận diện được
    detected_class_ids = result.boxes.cls.cpu().numpy()
    
    # Đếm số lượng từng loại
    counts = {}
    for cls_id in detected_class_ids:
        class_name = names_dict[int(cls_id)]
        counts[class_name] = counts.get(class_name, 0) + 1

    # Tính độ tin cậy trung bình của toàn bộ các box
    avg_conf = result.boxes.conf.cpu().numpy().mean() * 100

    # 4. Lập báo cáo trả về giao diện
    report = f"✅ Báo cáo quét hoàn tất (Ngưỡng tin cậy: {conf_threshold})\n"
    report += f"🎯 Độ tin cậy trung bình toàn khung hình: {avg_conf:.2f}%\n"
    report += "-" * 30 + "\n"
    report += f"🍎 Tổng số lượng phát hiện: {len(detected_class_ids)} thực thể\n"
    
    for name, count in counts.items():
        report += f"   ➤ {name}: {count}\n"

    return final_image, report

import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def process_faster_rcnn_prediction(image, model, conf_threshold=0.5):
    """
    Khai thác trực tiếp mô hình PyTorch thuần để dự đoán và tự động vẽ Bounding Box.
    """
    # --- 1. TIỀN XỬ LÝ (PRE-PROCESSING) ---
    # Chuyển đổi ảnh PIL sang Tensor và scale về [0, 1]
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0) # Thêm chiều batch (1, C, H, W)

    # --- 2. CHẠY INFERENCE ---
    with torch.no_grad():
        prediction = model(img_tensor)[0] # Lấy kết quả của ảnh đầu tiên

    # --- 3. HẬU XỬ LÝ VÀ KẾT XUẤT (POST-PROCESSING) ---
    # Chuyển ảnh PIL về OpenCV (BGR) để vẽ tay
    img_cv2 = np.array(image)
    img_cv2 = img_cv2[:, :, ::-1].copy() 

    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()

    # Ánh xạ ID (Cần khớp với file annotation JSON của bạn)
    class_map = {1: 'Green_Apple', 2: 'Red_Apple'}
    
    counts = {}
    valid_detections = 0
    total_conf = 0.0

    # Lọc qua các dự đoán và vẽ box
    for box, score, label in zip(boxes, scores, labels):
        if score >= conf_threshold:
            valid_detections += 1
            total_conf += score
            class_name = class_map.get(int(label), f"Unknown_{label}")
            counts[class_name] = counts.get(class_name, 0) + 1
            
            # Tọa độ box
            x1, y1, x2, y2 = map(int, box)
            
            # Vẽ HCN và nhãn (Xanh lá)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{class_name} {score:.2f}"
            cv2.putText(img_cv2, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Đảo lại hệ màu RGB cho Gradio
    final_image = Image.fromarray(img_cv2[:, :, ::-1])

    # --- 4. THỐNG KÊ CHI TIẾT ---
    if valid_detections == 0:
        return final_image, "⚠️ Không phát hiện quả táo nào (Faster R-CNN)."

    avg_conf = (total_conf / valid_detections) * 100
    report = f"✅ Báo cáo quét Faster R-CNN (Ngưỡng tin cậy: {conf_threshold})\n"
    report += f"🎯 Độ tin cậy trung bình: {avg_conf:.2f}%\n"
    report += "-" * 30 + "\n"
    report += f"🍎 Tổng số lượng phát hiện: {valid_detections} thực thể\n"
    
    for name, count in counts.items():
        report += f"   ➤ {name}: {count}\n"

    return final_image, report

import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import torch

def process_custom_retinanet(image, model, conf_threshold=0.35):
    """
    Xử lý tiền kỳ và hậu kỳ cho CSPDarknet + PANet + CBAM + RetinaNet[cite: 1]
    """
    # 1. Preprocessing (Resize về 640x640)[cite: 1]
    orig_w, orig_h = image.size
    resized = image.resize((640, 640), Image.BILINEAR)
    tensor = T.functional.to_tensor(resized).unsqueeze(0)
    
    # 2. Chạy Inference
    with torch.no_grad():
        preds = model(tensor)[0]
    
    # 3. Hậu xử lý (Post-processing)
    mask = preds['scores'] >= conf_threshold
    boxes = preds['boxes'][mask].cpu().numpy()
    labels = preds['labels'][mask].cpu().numpy()
    scores = preds['scores'][mask].cpu().numpy()
    
    # Tính tỷ lệ scale ngược về ảnh gốc[cite: 1]
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0
    
    img_cv2 = np.array(image)
    img_cv2 = img_cv2[:, :, ::-1].copy() # Chuyển PIL RGB sang OpenCV BGR

    class_map = {1: 'Green_Apple', 2: 'Red_Apple'}
    counts = {}
    valid_detections = len(boxes)
    total_conf = 0.0

    for box, score, label in zip(boxes, scores, labels):
        total_conf += score
        class_name = class_map.get(int(label), f"Unknown_{label}")
        counts[class_name] = counts.get(class_name, 0) + 1
        
        # Scale Tọa độ Box[cite: 1]
        x1, y1, x2, y2 = box
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        
        # Vẽ Box
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv2, f"{class_name} {score:.2f}", (x1, max(y1 - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    final_image = Image.fromarray(img_cv2[:, :, ::-1])

    if valid_detections == 0:
        return final_image, "⚠️ Không phát hiện quả táo nào."

    avg_conf = (total_conf / valid_detections) * 100
    report = f"✅ Báo cáo quét Custom RetinaNet\n🎯 Độ tin cậy trung bình: {avg_conf:.2f}%\n"
    report += "-" * 30 + f"\n🍎 Tổng phát hiện: {valid_detections} thực thể\n"
    for name, count in counts.items():
        report += f"   ➤ {name}: {count}\n"

    return final_image, report