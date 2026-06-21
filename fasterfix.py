import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# ⚠️ QUAN TRỌNG: Import các hàm từ file code của nhóm bạn!
# Giả sử chúng nằm trong file utils.py
from utils import build_dataloaders, compute_confusion_matrix, compute_map_coco

def evaluate_faster_rcnn_accuracy(model_path):
    print("=" * 60)
    print(f"🚀 BẮT ĐẦU ĐO ĐỘ CHÍNH XÁC: {model_path}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Nạp Dataset (Tập Test)
    print("[1/3] Đang nạp tập dữ liệu Test...")
    try:
        # Gọi hàm tạo data loader của bạn (có thể cần sửa tên hàm cho khớp)
        _, test_loader = build_dataloaders() 
    except Exception as e:
        print(f"❌ Lỗi nạp Dataset. Hãy kiểm tra lại hàm build_dataloaders của bạn: {e}")
        return

    # 2. Khởi tạo và nạp Model
    print("[2/3] Đang nạp cấu trúc Faster R-CNN...")
    backbone = resnet_fpn_backbone(backbone_name='resnet18', weights=None)
    model = FasterRCNN(backbone, num_classes=3) # 3 class: Background, Green, Red
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Bắt buộc phải khóa model

    # 3. Chấm điểm
    print("[3/3] Đang quét tập Test (Quá trình này mất vài phút trên CPU)...")
    with torch.no_grad():
        # Lấy Confusion Matrix để tính Precision & Recall
        tp, fp, fn = compute_confusion_matrix(model, test_loader, device)

        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())

        precision = (total_tp / (total_tp + total_fp)) * 100 if (total_tp + total_fp) > 0 else 0
        recall = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0

        # Lấy mAP
        map_coco, per_cls_coco, map50, _ = compute_map_coco(model, test_loader, device)

    # 4. In ra Bảng số liệu cuối cùng
    print("\n" + "=" * 50)
    print("🎯 BẢNG SỐ LIỆU ĐỘ CHÍNH XÁC (FASTER R-CNN):")
    print("=" * 50)
    print(f" ➤ mAP50 (%)    : {map50 * 100:.2f}")
    print(f" ➤ mAP50:95 (%) : {map_coco * 100:.2f}")
    print(f" ➤ Precision (%)  : {precision:.2f}")
    print(f" ➤ Recall (%)   : {recall:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    # Điền đường dẫn file .pth của bạn vào đây
    evaluate_faster_rcnn_accuracy("models/faster_rcnn_resnet18.pth")