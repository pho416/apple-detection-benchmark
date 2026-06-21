import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm # Thêm thư viện này để hiển thị thanh tiến trình cho đỡ sốt ruột

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as T
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CẤU HÌNH & HẰNG SỐ 
# ==========================================
TEST_IMG_DIR = "E:/Temp/demo_code/dataset/images/test"
TEST_LABEL_DIR = "E:/Temp/demo_code/dataset/labels/test"
WEIGHTS_PATH = "E:/Temp/demo_code/models/faster_rcnn_resnet18.pth" 

LABEL_MAP = {0: "background", 1: "Green_Apple", 2: "Red_Apple"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET ĐỌC FORMAT YOLO (.txt)
# ==========================================
class YoloToFasterRCNNDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        valid_extensions = ('.jpg', '.jpeg', '.png')
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_path = self.labels_dir / label_name

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        x_center, y_center, box_w, box_h = map(float, parts[1:5])
                        
                        x1 = (x_center - box_w / 2) * img_w
                        y1 = (y_center - box_h / 2) * img_h
                        x2 = (x_center + box_w / 2) * img_w
                        y2 = (y_center + box_h / 2) * img_h

                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_id + 1)

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

class ToTensor:
    def __call__(self, image, target):
        # Chỉ ép kiểu ảnh thành Tensor, bỏ qua và giữ nguyên target
        image = T.functional.to_tensor(image)
        return image, target

# ==========================================
# 3. CÁC HÀM ĐÁNH GIÁ (ĐÃ TỐI ƯU HÓA)
# ==========================================
def get_all_predictions(model, loader, device):
    """Chạy dự đoán ĐÚNG 1 LẦN và lưu vào RAM"""
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\n⏳ Đang quét ảnh qua AI Model (Chỉ 1 lần duy nhất)...")
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Inference"):
            images = [img.to(device) for img in images]
            preds = model(images)
            
            # Kéo dữ liệu về lại CPU để giải phóng VRAM của card đồ họa
            for p in preds:
                all_preds.append({k: v.cpu() for k, v in p.items()})
            for t in targets:
                all_targets.append({k: v.cpu() for k, v in t.items()})
                
    return all_preds, all_targets

def compute_map_simple(all_preds, all_targets, iou_threshold=0.50):
    class_preds = {c: [] for c in LABEL_MAP if c > 0}
    class_gts = {c: 0 for c in LABEL_MAP if c > 0}

    for pred, gt in zip(all_preds, all_targets):
        gt_boxes, gt_labels = gt["boxes"], gt["labels"]
        pred_boxes, pred_scores, pred_labels = pred["boxes"], pred["scores"], pred["labels"]

        for lbl in gt_labels:
            if lbl.item() in class_gts: class_gts[lbl.item()] += 1

        matched_gt = set()
        for i in range(len(pred_boxes)):
            sc, lbl = pred_scores[i].item(), pred_labels[i].item()
            if lbl not in class_preds: continue

            gt_mask = gt_labels == lbl
            is_tp = 0
            if gt_mask.any():
                gt_b = gt_boxes[gt_mask]
                ious = box_iou(pred_boxes[i].unsqueeze(0), gt_b)[0]
                best_iou, best_j = ious.max(0)
                if best_iou.item() >= iou_threshold and best_j.item() not in matched_gt:
                    is_tp = 1
                    matched_gt.add(best_j.item())
            class_preds[lbl].append((sc, is_tp))

    aps = {}
    for cls_id, preds_list in class_preds.items():
        n_gt = class_gts[cls_id]
        if n_gt == 0 or len(preds_list) == 0:
            aps[cls_id] = 0.0
            continue
        preds_list.sort(key=lambda x: -x[0])
        tp_cum = fp_cum = 0
        precisions, recalls = [], []
        for sc, is_tp in preds_list:
            if is_tp: tp_cum += 1
            else: fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt)
            
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_t = [precisions[i] for i, r in enumerate(recalls) if r >= t]
            ap += (max(prec_at_t) if prec_at_t else 0.0) / 11.0
        aps[cls_id] = ap

    return float(np.mean(list(aps.values()))), aps

def compute_map_coco(all_preds, all_targets):
    iou_thresholds = [round(0.50 + 0.05 * k, 2) for k in range(10)]
    ap_at_thresh = {}
    for thr in iou_thresholds:
        _, aps = compute_map_simple(all_preds, all_targets, iou_threshold=thr)
        ap_at_thresh[thr] = aps

    per_class_coco = {}
    for cls_id in [1, 2]:
        per_class_coco[cls_id] = float(np.mean([ap_at_thresh[thr][cls_id] for thr in iou_thresholds]))

    return float(np.mean(list(per_class_coco.values()))), per_class_coco

def compute_confusion_matrix(all_preds, all_targets, iou_threshold=0.5, score_threshold=0.5):
    tp, fp, fn = {1: 0, 2: 0}, {1: 0, 2: 0}, {1: 0, 2: 0}

    for pred, gt in zip(all_preds, all_targets):
        gt_boxes, gt_labels = gt["boxes"], gt["labels"]
        
        mask = pred["scores"] >= score_threshold
        pred_boxes = pred["boxes"][mask]
        pred_labels = pred["labels"][mask]

        for cls_id in [1, 2]:
            gt_b = gt_boxes[gt_labels == cls_id]
            pr_b = pred_boxes[pred_labels == cls_id]
            fn[cls_id] += len(gt_b)

            if len(pr_b) == 0: continue
            if len(gt_b) == 0:
                fp[cls_id] += len(pr_b)
                continue

            ious = box_iou(pr_b, gt_b)
            matched_gt = set()
            for i in range(len(pr_b)):
                best_iou, best_j = ious[i].max(0)
                if best_iou.item() >= iou_threshold and best_j.item() not in matched_gt:
                    tp[cls_id] += 1
                    fn[cls_id] -= 1
                    matched_gt.add(best_j.item())
                else:
                    fp[cls_id] += 1

    print("\n" + "="*65)
    print(f"  CONFUSION MATRIX  (IoU >= {iou_threshold}, Score >= {score_threshold})")
    print("="*65)
    print(f"  {'Class':<15} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Precision':>10} | {'Recall':>8}")
    print("  " + "-"*60)
    for cls_id in [1, 2]:
        name = LABEL_MAP[cls_id]
        prec = tp[cls_id] / (tp[cls_id] + fp[cls_id]) if (tp[cls_id] + fp[cls_id]) > 0 else 0
        rec = tp[cls_id] / (tp[cls_id] + fn[cls_id]) if (tp[cls_id] + fn[cls_id]) > 0 else 0
        print(f"  {name:<15} | {tp[cls_id]:>6} | {fp[cls_id]:>6} | {fn[cls_id]:>6} | {prec:>10.4f} | {rec:>8.4f}")

# ==========================================
# 4. LUỒNG CHÍNH (MAIN)
# ==========================================
if __name__ == "__main__":
    print(f"🖥️  Hệ thống đang chạy trên: {DEVICE.type.upper()}")
    if DEVICE.type == "cpu":
        print("⚠️ CẢNH BÁO: Máy bạn đang chạy bằng CPU nên thời gian quét ảnh sẽ khá lâu!")

    print("🚀 Đang khởi tạo Model Faster R-CNN...")
    backbone = resnet_fpn_backbone('resnet18', pretrained=False)
    model_faster_rcnn = FasterRCNN(backbone, num_classes=3)
    
    try:
        model_faster_rcnn.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        model_faster_rcnn.to(DEVICE)
        model_faster_rcnn.eval() 
        print("✅ Đã nạp thành công trọng số Faster R-CNN")
    except Exception as e:
        print(f"❌ Lỗi nạp model: {e}")
        exit()

    print("🚀 Đang chuẩn bị Test Dataset từ YOLO Format...")
    test_transform = Compose([ToTensor()])
    test_ds = YoloToFasterRCNNDataset(TEST_IMG_DIR, TEST_LABEL_DIR, transforms=test_transform)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0) # Tăng batch_size lên 4

    print("\n" + "="*50)
    print(f"BẮT ĐẦU ĐÁNH GIÁ ({len(test_ds)} ảnh test)")
    print("="*50)

    # BƯỚC QUAN TRỌNG NHẤT: Chạy suy luận 1 lần và lưu lại
    all_preds, all_targets = get_all_predictions(model_faster_rcnn, test_loader, DEVICE)

    print("\n🧮 Đang tính toán các chỉ số mAP...")
    # Truyền dữ liệu đã lưu vào các hàm tính toán
    map50, aps50 = compute_map_simple(all_preds, all_targets, iou_threshold=0.50)
    map75, aps75 = compute_map_simple(all_preds, all_targets, iou_threshold=0.75)
    print(f"\n  mAP@0.50      = {map50:.4f}")
    print(f"  mAP@0.75      = {map75:.4f}")

    map_coco, per_cls_coco = compute_map_coco(all_preds, all_targets)
    print(f"  mAP@0.50:0.95 = {map_coco:.4f}  (COCO Standard)")

    print(f"\n  {'Class':<16} {'AP@0.5':>8} {'AP@0.75':>9} {'AP_COCO':>9}")
    print("  " + "-"*44)
    for cls_id in aps50:
        name = LABEL_MAP[cls_id]
        print(f"  {name:<16} {aps50[cls_id]:>8.4f} {aps75.get(cls_id,0):>9.4f} {per_cls_coco.get(cls_id,0):>9.4f}")

    compute_confusion_matrix(all_preds, all_targets, iou_threshold=0.5, score_threshold=0.5)