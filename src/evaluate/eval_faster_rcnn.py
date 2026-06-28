import os
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as T
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader

LABEL_MAP = {0: "background", 1: "Green_Apple", 2: "Red_Apple"}

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

        boxes, labels = [], []
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

def collate_fn(batch):
    return tuple(zip(*batch))

class ToTensor:
    def __call__(self, image, target):
        return T.functional.to_tensor(image), target

def measure_fps(model, device, imgsz=640, iterations=100):
    print("\n⏱️ Đang đo tốc độ khung hình (FPS)...")
    model.eval()
    dummy_img = [torch.zeros((3, imgsz, imgsz), dtype=torch.float32).to(device)]

    with torch.no_grad():
        for _ in range(10): model(dummy_img)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations): model(dummy_img)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    return iterations / total_time

def get_all_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    print("\n⏳ Đang quét ảnh qua AI Model (Chỉ 1 lần duy nhất)...")
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Inference"):
            images = [img.to(device) for img in images]
            preds = model(images)
            for p in preds: all_preds.append({k: v.cpu() for k, v in p.items()})
            for t in targets: all_targets.append({k: v.cpu() for k, v in t.items()})
    return all_preds, all_targets

def compute_map_simple(all_preds, all_targets, iou_threshold=0.50):
    class_preds = {c: [] for c in LABEL_MAP if c > 0}
    class_gts = {c: 0 for c in LABEL_MAP if c > 0}

    for pred, gt in zip(all_preds, all_targets):
        for lbl in gt["labels"]:
            if lbl.item() in class_gts: class_gts[lbl.item()] += 1

        matched_gt = set()
        for i in range(len(pred["boxes"])):
            sc, lbl = pred["scores"][i].item(), pred["labels"][i].item()
            if lbl not in class_preds: continue
            
            gt_mask = gt["labels"] == lbl
            is_tp = 0
            if gt_mask.any():
                gt_b = gt["boxes"][gt_mask]
                ious = box_iou(pred["boxes"][i].unsqueeze(0), gt_b)[0]
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
        tp_cum, fp_cum = 0, 0
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
    ap_at_thresh = {thr: compute_map_simple(all_preds, all_targets, iou_threshold=thr)[1] for thr in iou_thresholds}
    
    per_class_coco = {}
    for cls_id in [1, 2]:
        per_class_coco[cls_id] = float(np.mean([ap_at_thresh[thr][cls_id] for thr in iou_thresholds]))
    return float(np.mean(list(per_class_coco.values()))), per_class_coco

def compute_confusion_matrix(all_preds, all_targets, iou_threshold=0.5, score_threshold=0.5):
    tp, fp, fn = {1: 0, 2: 0}, {1: 0, 2: 0}, {1: 0, 2: 0}
    for pred, gt in zip(all_preds, all_targets):
        mask = pred["scores"] >= score_threshold
        pred_boxes, pred_labels = pred["boxes"][mask], pred["labels"][mask]
        
        for cls_id in [1, 2]:
            gt_b = gt["boxes"][gt["labels"] == cls_id]
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
    print(f"  CONFUSION MATRIX (IoU >= {iou_threshold}, Score >= {score_threshold})")
    print("="*65)
    print(f"  {'Class':<15} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Precision':>10} | {'Recall':>8}")
    print("  " + "-"*60)
    
    total_prec, total_rec = [], []
    for cls_id in [1, 2]:
        prec = tp[cls_id] / (tp[cls_id] + fp[cls_id]) if (tp[cls_id] + fp[cls_id]) > 0 else 0
        rec = tp[cls_id] / (tp[cls_id] + fn[cls_id]) if (tp[cls_id] + fn[cls_id]) > 0 else 0
        total_prec.append(prec); total_rec.append(rec)
        print(f"  {LABEL_MAP[cls_id]:<15} | {tp[cls_id]:>6} | {fp[cls_id]:>6} | {fn[cls_id]:>6} | {prec:>10.4f} | {rec:>8.4f}")
    return float(np.mean(total_prec)), float(np.mean(total_rec))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN Model")
    parser.add_argument("--img_dir", type=str, required=True, help="Thư mục chứa ảnh test")
    parser.add_argument("--label_dir", type=str, required=True, help="Thư mục chứa nhãn YOLO (.txt)")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file trọng số (.pth)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Hệ thống đang chạy trên: {DEVICE.type.upper()}")

    print("🚀 Đang khởi tạo Model Faster R-CNN...")
    backbone = resnet_fpn_backbone('resnet18', weights=None)
    model = FasterRCNN(backbone, num_classes=3)
    
    try:
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() 
        print(f"✅ Đã nạp thành công trọng số Faster R-CNN")
    except Exception as e:
        print(f"❌ Lỗi nạp model: {e}")
        exit()

    test_ds = YoloToFasterRCNNDataset(args.img_dir, args.label_dir, transforms=ToTensor())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    fps_score = measure_fps(model, DEVICE)

    print(f"\nBẮT ĐẦU ĐÁNH GIÁ ({len(test_ds)} ảnh test)")
    all_preds, all_targets = get_all_predictions(model, test_loader, DEVICE)

    map50, aps50 = compute_map_simple(all_preds, all_targets, iou_threshold=0.50)
    map75, aps75 = compute_map_simple(all_preds, all_targets, iou_threshold=0.75)
    map_coco, per_cls_coco = compute_map_coco(all_preds, all_targets)
    mean_prec, mean_rec = compute_confusion_matrix(all_preds, all_targets)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ BENCHMARK: Faster R-CNN (ResNet18)")
    print("=" * 60)
    print(f"mAP50 (%)    : {map50 * 100:.2f}")
    print(f"mAP50:95 (%) : {map_coco * 100:.2f}")
    print(f"Precision (%)  : {mean_prec * 100:.2f}")
    print(f"Recall (%)   : {mean_rec * 100:.2f}")
    print(f"Params (M)   : {params_m:.2f} M")
    print(f"FPS          : {fps_score:.2f}")
    print("=" * 60)