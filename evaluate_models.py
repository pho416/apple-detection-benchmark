from ultralytics import YOLO
from ultralytics.models.rtdetr import RTDETR
import torch
import time

def benchmark_ultralytics(model_path, data_yaml):
    print(f"\n🚀 Đang đánh giá model: {model_path}")
    
    # 1. Tải mô hình
    if "yolo" in model_path.lower():
        model = YOLO(model_path)
    else:
        model = RTDETR(model_path)
        
    # 2. Lấy Params (M) và GFLOPs (Độ nặng mô hình)
    # Hàm info() trả về tuple: (số layer, params, gradients, flops)
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    
    # Thử tính GFLOPs bằng hàm chuyên dụng của Ultralytics
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = get_flops(model.model, imgsz=640)
    except Exception:
        gflops = 0.0
    
    # 3. Lấy mAP, Precision, Recall (Độ chính xác)
    print("Đang quét tập Test (Vui lòng đợi)...")
    # Thay 'test' bằng 'val' nếu file yaml của bạn đặt tên tập đánh giá là val
    metrics = model.val(data=data_yaml, split='test', verbose=False)
    
    map50 = metrics.box.map50 * 100
    map50_95 = metrics.box.map * 100
    precision = metrics.box.p.mean() * 100
    recall = metrics.box.r.mean() * 100
    
    # 4. Đo FPS (Tốc độ)
    print("Đang đo FPS...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_img = torch.zeros((1, 3, 640, 640)).to(device)
    model.to(device)
    
    # Làm nóng GPU (Warm-up)
    for _ in range(10): 
        model.predict(dummy_img, verbose=False)
        
    # Đo thời gian 100 vòng lặp
    start = time.time()
    for _ in range(100): 
        model.predict(dummy_img, verbose=False)
    total_time = time.time() - start
    fps = 100 / total_time
    
    # 5. IN RA KẾT QUẢ ĐỂ BẠN ĐIỀN VÀO BẢNG
    print("=" * 60)
    print(f"KẾT QUẢ BENCHMARK: {model_path}")
    print("=" * 60)
    print(f"mAP50 (%)    : {map50:.2f}")
    print(f"mAP50:95 (%) : {map50_95:.2f}")
    print(f"Precision (%)  : {precision:.2f}")
    print(f"Recall (%)   : {recall:.2f}")
    print(f"Params (M)   : {params_m:.2f} M")
    print(f"GFLOPs       : {gflops:.2f}")
    print(f"FPS          : {fps:.2f}")
    print("=" * 60)

# Chạy thử cho model của bạn:
benchmark_ultralytics("models/yolo.pt", "dataset/data.yaml")