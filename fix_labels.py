import os
import glob

# Trỏ chính xác vào thư mục chứa nhãn của tập test
# Hãy sửa đường dẫn này cho khớp với thư mục thực tế của bạn
label_dir = r"E:\Temp\demo_code\dataset\labels\test"

# Lấy danh sách toàn bộ file .txt
txt_files = glob.glob(os.path.join(label_dir, "*.txt"))

print(f"🔍 Tìm thấy {len(txt_files)} file nhãn cần xử lý...")

count = 0
for file_path in txt_files:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            # Nếu nhãn đang là 0 (Xanh), ép nó thành 1 (Đỏ)
            if parts[0] == '0':
                parts[0] = '1'
                count += 1
            new_lines.append(' '.join(parts) + '\n')
            
    # Ghi đè lại file
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

print(f"✅ Đã biến đổi thành công {count} dòng nhãn từ 0 sang 1!")