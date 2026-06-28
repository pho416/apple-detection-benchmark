import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def process_apple_prediction(image, model, conf_threshold=0.4):
    """Xử lý Inference cho họ Ultralytics (YOLO, MSRRT)"""
    results = model.predict(source=image, imgsz=640, conf=conf_threshold)
    result = results[0] 

    annotated_img_bgr = result.plot()
    annotated_img_rgb = annotated_img_bgr[..., ::-1] 
    final_image = Image.fromarray(annotated_img_rgb)

    names_dict = result.names 
    
    if result.boxes is None or len(result.boxes) == 0:
        return final_image, "⚠️ Không phát hiện quả táo nào trong khung hình."

    detected_class_ids = result.boxes.cls.cpu().numpy()
    
    counts = {}
    for cls_id in detected_class_ids:
        class_name = names_dict[int(cls_id)]
        counts[class_name] = counts.get(class_name, 0) + 1

    avg_conf = result.boxes.conf.cpu().numpy().mean() * 100

    report = f"✅ Báo cáo quét hoàn tất (Ngưỡng tin cậy: {conf_threshold})\n"
    report += f"🎯 Độ tin cậy trung bình: {avg_conf:.2f}%\n"
    report += "-" * 30 + "\n"
    report += f"🍎 Tổng số lượng phát hiện: {len(detected_class_ids)} thực thể\n"
    for name, count in counts.items():
        report += f"   ➤ {name}: {count}\n"

    return final_image, report

def process_faster_rcnn_prediction(image, model, device, conf_threshold=0.5):
    """Xử lý Inference thuần PyTorch cho Faster R-CNN"""
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0] 

    img_cv2 = np.array(image)
    img_cv2 = img_cv2[:, :, ::-1].copy() 

    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()

    class_map = {1: 'Green_Apple', 2: 'Red_Apple'}
    counts = {}
    valid_detections = 0
    total_conf = 0.0

    for box, score, label in zip(boxes, scores, labels):
        if score >= conf_threshold:
            valid_detections += 1
            total_conf += score
            class_name = class_map.get(int(label), f"Unknown_{label}")
            counts[class_name] = counts.get(class_name, 0) + 1
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv2, f"{class_name} {score:.2f}", (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    final_image = Image.fromarray(img_cv2[:, :, ::-1])

    if valid_detections == 0:
        return final_image, "⚠️ Không phát hiện quả táo nào (Faster R-CNN)."

    avg_conf = (total_conf / valid_detections) * 100
    report = f"✅ Báo cáo quét Faster R-CNN\n🎯 Độ tin cậy trung bình: {avg_conf:.2f}%\n"
    report += "-" * 30 + f"\n🍎 Tổng phát hiện: {valid_detections} thực thể\n"
    for name, count in counts.items():
        report += f"   ➤ {name}: {count}\n"

    return final_image, report

def process_custom_retinanet(image, model, device, conf_threshold=0.35):
    """Xử lý Inference cho Custom RetinaNet"""
    orig_w, orig_h = image.size
    resized = image.resize((640, 640), Image.BILINEAR)
    tensor = T.functional.to_tensor(resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(tensor)[0]
    
    mask = preds['scores'] >= conf_threshold
    boxes = preds['boxes'][mask].cpu().numpy()
    labels = preds['labels'][mask].cpu().numpy()
    scores = preds['scores'][mask].cpu().numpy()
    
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0
    
    img_cv2 = np.array(image)
    img_cv2 = img_cv2[:, :, ::-1].copy()

    class_map = {1: 'Green_Apple', 2: 'Red_Apple'}
    counts = {}
    valid_detections = len(boxes)
    total_conf = 0.0

    for box, score, label in zip(boxes, scores, labels):
        total_conf += score
        class_name = class_map.get(int(label), f"Unknown_{label}")
        counts[class_name] = counts.get(class_name, 0) + 1
        
        x1, y1, x2, y2 = box
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        
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