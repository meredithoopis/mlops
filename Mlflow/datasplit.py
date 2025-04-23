import os
import random
import shutil

# Đường dẫn thư mục gốc
image_dir = "images"
label_dir = "labels/train"

# Các thư mục đích
train_image_dir = "images/train"
val_image_dir = "images/val"
test_image_dir = "images/test"

train_label_dir = "labels/train"
val_label_dir = "labels/val"
test_label_dir = "labels/test"

# Tạo các thư mục nếu chưa có
for d in [train_image_dir, val_image_dir, test_image_dir, train_label_dir, val_label_dir, test_label_dir]:
    os.makedirs(d, exist_ok=True)

# Lấy danh sách file ảnh (không có phần mở rộng)
image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Shuffle và chia tỉ lệ
random.seed(42)
random.shuffle(image_files)

total = len(image_files)
train_count = int(0.8 * total)
val_count = int(0.1 * total)

train_samples = image_files[:train_count]
val_samples = image_files[train_count:train_count + val_count]
test_samples = image_files[train_count + val_count:]

# Hàm hỗ trợ di chuyển ảnh + label
def move(name_list, img_dst, lbl_dst):
    for name in name_list:
        img_src = os.path.join(image_dir, name + ".jpg")
        lbl_src = os.path.join(label_dir, name + ".txt")
        img_dst_path = os.path.join(img_dst, name + ".jpg")
        lbl_dst_path = os.path.join(lbl_dst, name + ".txt")

        if os.path.exists(img_src):
            shutil.move(img_src, img_dst_path)
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst_path)

# Di chuyển file
move(train_samples, train_image_dir, train_label_dir)
move(val_samples, val_image_dir, val_label_dir)
move(test_samples, test_image_dir, test_label_dir)