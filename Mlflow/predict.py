from ultralytics import YOLO
import os

# Load model đã huấn luyện
model = YOLO("/mlartifacts/0/16fa8bd85843416f930b613acbdd3b58/artifacts/weights/best.pt")

# Đường dẫn thư mục test
test_folder = "/home/laplace/project-yolo/images/test"

# Tạo folder output
save_dir = "runs/detect/test_result"
os.makedirs(save_dir, exist_ok=True)

# Dự đoán và lưu ảnh chứa bounding box
model.predict(
    source=test_folder,
    save=True,
    save_txt=True,   # lưu cả kết quả bounding box dạng txt hay ko
    project="runs/detect", 
    name="test_result",
    exist_ok=True,
    conf=0.25        # ngưỡng confidence
)
