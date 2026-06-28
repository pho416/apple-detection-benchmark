import os
import sys
import gradio as gr
from PIL import Image
import torch

# ==========================================
# 0. THIẾT LẬP ĐƯỜNG DẪN TỰ ĐỘNG
# ==========================================
# Lấy đường dẫn tuyệt đối của thư mục gốc project (apple_detection_project)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "models")

from ultralytics.models.rtdetr import RTDETR
from ultralytics.models.yolo import YOLO
from src.models.custom_retinanet import build_model  # Import chuẩn từ src/models/
from utils import process_apple_prediction, process_faster_rcnn_prediction, process_custom_retinanet

# ==========================================
# 1. NẠP THỂ XÁC VÀ LINH HỒN LÊN RAM
# ==========================================
print("Đang khởi động hệ thống nhận diện...")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Hệ thống sẽ chạy web bằng: {DEVICE.type.upper()}")

# --- Load Model 1: MSRRT-DERT (.pt) ---
try:
    model_msrrt = RTDETR(os.path.join(MODEL_DIR, "MSRRT-DERT.pt")) 
    print("✅ Đã nạp MSRRT-DERT")
except Exception as e:
    print(f"❌ Lỗi nạp MSRRT-DERT: {e}")

# --- Load Model 2: Custom RetinaNet ---
try:
    model_custom_retina = build_model(num_classes=3)
    model_custom_retina.load_state_dict(torch.load(os.path.join(MODEL_DIR, "cspdarknet_panet_cbam_retinanet.pth"), map_location=DEVICE))
    model_custom_retina.eval() 
    model_custom_retina.to(DEVICE)
    print("✅ Đã nạp Custom RetinaNet (CSPDarknet + PANet + CBAM)")
except Exception as e:
    print(f"❌ Lỗi nạp Custom RetinaNet: {e}")

# --- Load Model 3: YOLO ---
try:
    model_yolo = YOLO(os.path.join(MODEL_DIR, "yolo.pt"))
    print("✅ Đã nạp YOLO")
except Exception as e:
    print(f"❌ Lỗi nạp YOLO: {e}")

# --- Load Model 4: Faster R-CNN (.pth) ---
try:
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    
    backbone = resnet_fpn_backbone('resnet18', weights=None)
    model_faster_rcnn = FasterRCNN(backbone, num_classes=3)
    
    model_faster_rcnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "faster_rcnn_resnet18.pth"), map_location=DEVICE))
    model_faster_rcnn.eval()
    model_faster_rcnn.to(DEVICE)
    print("✅ Đã nạp Faster R-CNN")
except Exception as e:
    print(f"❌ Lỗi nạp Faster R-CNN: {e}")

print("🚀 HỆ THỐNG ĐÃ SẴN SÀNG!")

# ==========================================
# 2. HÀM ĐỊNH TUYẾN
# ==========================================
def predict_apple(image, model_name, conf_threshold):
    if image is None:
        return "Vui lòng tải ảnh lên!", None

    try:
        if model_name == "MSRRT-DERT":
            final_image, report_text = process_apple_prediction(image, model_msrrt, conf_threshold)
        elif model_name == "YOLO":
            final_image, report_text = process_apple_prediction(image, model_yolo, conf_threshold)
        elif model_name == "Custom RetinaNet":
            final_image, report_text = process_custom_retinanet(image, model_custom_retina, DEVICE, conf_threshold)
        elif model_name == "Faster R-CNN (ResNet18)":
            final_image, report_text = process_faster_rcnn_prediction(image, model_faster_rcnn, DEVICE, conf_threshold)
        else:
            return "⚠️ Model này chưa được tích hợp!", image
            
        return report_text, final_image
    except Exception as e:
        return f"❌ Hệ thống gặp sự cố: {str(e)}", image

# ==========================================
# 3. GIAO DIỆN UI TỔNG THỂ
# ==========================================
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
        # 🍎 Hệ thống Nhận diện và Phân loại Táo
        **Đồ án môn Thị giác máy tính**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Cấu hình & Đầu vào")
            model_dropdown = gr.Dropdown(
                choices=["MSRRT-DERT", "YOLO", "Custom RetinaNet", "Faster R-CNN (ResNet18)"],
                label="Chọn Model để đánh giá",
                value="MSRRT-DERT"
            )
            conf_slider = gr.Slider(minimum=0.1, maximum=0.95, value=0.4, step=0.05, label="Ngưỡng tin cậy (Confidence Threshold)")
            input_image = gr.Image(label="Tải ảnh quả táo lên", type="pil")
            predict_btn = gr.Button("🚀 Khởi chạy quét", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Kết quả phân tích")
            output_text = gr.Textbox(label="Thông số trích xuất", lines=6)
            output_image = gr.Image(label="Bản đồ Bounding Box")

    predict_btn.click(
        fn=predict_apple,
        inputs=[input_image, model_dropdown, conf_slider],
        outputs=[output_text, output_image]
    )

if __name__ == "__main__":
    demo.launch(share=True)