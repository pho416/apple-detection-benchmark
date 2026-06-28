import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) # Dòng này quan trọng!

import yaml
import torch
import time
import argparse
from ultralytics import YOLO
from ultralytics.models.rtdetr import RTDETR

NAMES_MAP = {0: "Green_Apple", 1: "Red_Apple"}

def benchmark_ultralytics(weights_path, img_dir, imgsz=640):
    print(f"\n🚀 Đang đánh giá model: {os.path.basename(weights_path)}")

    # 1. Tạo file YAML tạm thời để "ép" Ultralytics đọc thẳng từ thư mục test
    temp_yaml_path = "temp_eval_dataset.yaml"
    yaml_content = {
        "train": img_dir,
        "val": img_dir,
        "test": img_dir,
        "names": NAMES_MAP
    }
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)

    try:
        # 2. Khởi tạo mô hình
        if "yolo" in weights_path.lower():
            model = YOLO(weights_path)
        else:
            model = RTDETR(weights_path)

        # 3. Tính toán số lượng tham số và GFLOPs
        params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
        try:
            from ultralytics.utils.torch_utils import get_flops
            gflops = get_flops(model.model, imgsz=imgsz)
        except Exception:
            gflops = 0.0

        # 4. Quét tập đánh giá (mAP, Precision, Recall)
        print("Đang quét tập Test (Vui lòng đợi)...")
        metrics = model.val(data=temp_yaml_path, split='test', verbose=False, imgsz=imgsz)

        map50 = metrics.box.map50 * 100
        map50_95 = metrics.box.map * 100
        precision = metrics.box.p.mean() * 100
        recall = metrics.box.r.mean() * 100

        # 5. Đo FPS
        print("⏱️ Đang đo FPS...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dummy_img = torch.zeros((1, 3, imgsz, imgsz)).to(device)
        model.to(device)

        for _ in range(10): # Warm-up
            model.predict(dummy_img, verbose=False)

        if device == 'cuda': torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            model.predict(dummy_img, verbose=False)
        if device == 'cuda': torch.cuda.synchronize()
        
        total_time = time.time() - start
        fps = 100 / total_time

        # 6. In kết quả
        print("\n" + "=" * 60)
        print(f"KẾT QUẢ BENCHMARK: {os.path.basename(weights_path)}")
        print("=" * 60)
        print(f"mAP50 (%)    : {map50:.2f}")
        print(f"mAP50:95 (%) : {map50_95:.2f}")
        print(f"Precision (%)  : {precision:.2f}")
        print(f"Recall (%)   : {recall:.2f}")
        print(f"Params (M)   : {params_m:.2f} M")
        print(f"GFLOPs       : {gflops:.2f}")
        print(f"FPS          : {fps:.2f}")
        print("=" * 60)

    finally:
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ultralytics Models (YOLO, RTDETR, etc.)")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn đến file weights (.pt)")
    parser.add_argument("--img_dir", type=str, required=True, help="Đường dẫn đến thư mục chứa ảnh test (labels phải nằm ngang hàng hoặc chung thư mục theo chuẩn YOLO)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (default: 640)")
    args = parser.parse_args()

    benchmark_ultralytics(args.weights, args.img_dir, args.imgsz)