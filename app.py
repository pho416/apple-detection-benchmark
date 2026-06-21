import gradio as gr
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ultralytics.models.rtdetr import RTDETR
from ultralytics.models.yolo import YOLO
from custom_retinanet import build_model
from utils import process_apple_prediction, process_faster_rcnn_prediction, process_custom_retinanet

# ==========================================
# 1. NẠP THỂ XÁC VÀ LINH HỒN LÊN RAM
# ==========================================
print("Đang khởi động hệ thống nhận diện...")

# --- Load Model 1: MSRRT-DERT (.pt) ---
try:
    model_msrrt = RTDETR("models/MSRRT-DERT.pt") 
    print("✅ Đã nạp MSRRT-DERT")
except Exception as e:
    print(f"❌ Lỗi nạp MSRRT-DERT: {e}")


try:
    # Khởi tạo thể xác từ class bạn định nghĩa
    model_custom_retina = build_model(num_classes=3)
    # Bơm linh hồn
    model_custom_retina.load_state_dict(torch.load("models/cspdarknet_panet_cbam_retinanet.pth", map_location=torch.device('cpu')))
    model_custom_retina.eval() 
    print("✅ Đã nạp Custom RetinaNet (CSPDarknet + PANet + CBAM)")
except Exception as e:
    print(f"❌ Lỗi nạp Custom RetinaNet: {e}")

try:
    model_yolo = YOLO("models/yolo.pt")
    print("✅ Đã nạp YOLO")
except Exception as e:
    print(f"❌ Lỗi nạp YOLO: {e}")
# --- Load Model 2: Faster R-CNN (.pth) ---
try:
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    
    # Bước A: Khởi tạo chính xác Thể xác Faster R-CNN với backbone ResNet18
    # 1. Tạo backbone ResNet18
    backbone = resnet_fpn_backbone('resnet18', pretrained=False)
    # 2. Ráp backbone vào bộ khung Faster R-CNN
    model_faster_rcnn = FasterRCNN(backbone, num_classes=3)
    
    # Bước B: Bơm trọng số (Linh hồn)
    model_faster_rcnn.load_state_dict(torch.load("models/faster_rcnn_resnet18.pth", map_location=torch.device('cpu')))
    model_faster_rcnn.eval() # Bắt buộc phải khóa model ở chế độ đánh giá
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
        # Nhóm Ultralytics (Dùng chung 1 hàm xử lý)
        if model_name == "MSRRT-DERT":
            final_image, report_text = process_apple_prediction(image, model_msrrt, conf_threshold)

        elif model_name == "YOLO":
            final_image, report_text = process_apple_prediction(image, model_yolo, conf_threshold)
            
        # Nhóm Custom & Thuần PyTorch
        elif model_name == "Custom RetinaNet":
            final_image, report_text = process_custom_retinanet(image, model_custom_retina, conf_threshold)
        elif model_name == "Faster R-CNN (ResNet18)":
            final_image, report_text = process_faster_rcnn_prediction(image, model_faster_rcnn, conf_threshold)
            
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
                choices=[
                    "MSRRT-DERT", 
                    "YOLO",
                    "Custom RetinaNet",
                    "Faster R-CNN (ResNet18)"
                ],
                label="Chọn Model để đánh giá",
                value="MSRRT-DERT"
            )
            
            conf_slider = gr.Slider(
                minimum=0.1, 
                maximum=0.95, 
                value=0.4, 
                step=0.05, 
                label="Ngưỡng tin cậy (Confidence Threshold)"
            )
            
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