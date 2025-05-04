import os
import glob
import pandas as pd
import shutil
import random
from ultralytics import YOLO
import mlflow
from pathlib import Path

# Cấu hình đường dẫn
BASE_DIR = "/app"  # Trong Docker
DATA_DIR = os.path.join(BASE_DIR, "data_pipeline/flow")
DATASET_DIR = os.path.join(BASE_DIR, "tracking_pipeline/dataset")

# 1. Tạo cấu trúc thư mục
os.makedirs(os.path.join(DATASET_DIR, "images/train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "images/val"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "images/test"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "labels/val"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "labels/test"), exist_ok=True)

# 2. Kiểm tra dữ liệu
img_dir = os.path.join(DATA_DIR, "images")
label_file_path = os.path.join(DATA_DIR, "image.txt")

with open(label_file_path, 'r') as f:
    label_entries = [line.strip() for line in f.readlines()]
label_names = {entry.split('/')[-1] for entry in label_entries}
img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
img_names = {os.path.splitext(os.path.basename(f))[0] for f in img_files}

print(f"Num images: {len(img_files)}")
print(f"Num labels: {len(label_names)}")
print(f"Images with missing labels: {len(img_names - label_names)}")
print(f"Labels without image: {len(label_names - img_names)}")

# 3. Phân chia dataset
random.seed(1610)
random.shuffle(img_files)

train_split = int(len(img_files) * 0.7)
val_split = int(len(img_files) * 0.85)

train_files = img_files[:train_split]
val_files = img_files[train_split:val_split]
test_files = img_files[val_split:]

# Copy ảnh
for src_path in train_files:
    filename = os.path.basename(src_path)
    dst_path = os.path.join(DATASET_DIR, "images/train", filename)
    shutil.copy2(src_path, dst_path)
    
for src_path in val_files:
    filename = os.path.basename(src_path)
    dst_path = os.path.join(DATASET_DIR, "images/val", filename)
    shutil.copy2(src_path, dst_path)
    
for src_path in test_files:
    filename = os.path.basename(src_path)
    dst_path = os.path.join(DATASET_DIR, "images/test", filename)
    shutil.copy2(src_path, dst_path)

print(f"Split complete: {len(train_files)} training images, {len(val_files)} validation images, and {len(test_files)} test images")

# 4. Chuyển đổi nhãn sang định dạng YOLO
labels = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))
unique_labels = sorted(labels['LabelName_Text'].unique())
class_mapping = {label: idx for idx, label in enumerate(unique_labels)}

print("Class mapping:")
for label, class_id in class_mapping.items():
    print(f"  {label}: {class_id}")

train_images = {os.path.splitext(img)[0] for img in os.listdir(os.path.join(DATASET_DIR, "images/train")) if img.endswith(".jpg")}
val_images = {os.path.splitext(img)[0] for img in os.listdir(os.path.join(DATASET_DIR, "images/val")) if img.endswith(".jpg")}
test_images = {os.path.splitext(img)[0] for img in os.listdir(os.path.join(DATASET_DIR, "images/test")) if img.endswith(".jpg")}

for image_id, group in labels.groupby("ImageID"):
    if image_id in train_images:
        txt_path = os.path.join(DATASET_DIR, "labels/train", f"{image_id}.txt")
    elif image_id in val_images:
        txt_path = os.path.join(DATASET_DIR, "labels/val", f"{image_id}.txt")
    elif image_id in test_images:
        txt_path = os.path.join(DATASET_DIR, "labels/test", f"{image_id}.txt")
    else:
        continue
        
    with open(txt_path, "w") as f:
        for _, row in group.iterrows():
            class_id = class_mapping.get(row["LabelName_Text"], -1)
            if class_id == -1:
                continue
            x_center = (row["XMin"] + row["XMax"]) / 2
            y_center = (row["YMin"] + row["YMax"]) / 2
            width = row["XMax"] - row["XMin"]
            height = row["YMax"] - row["YMin"]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"Created YOLO labels in {DATASET_DIR}/labels/train, {DATASET_DIR}/labels/val, {DATASET_DIR}/labels/test")

# 5. Tạo file data.yaml
data_yaml = f"""
train: {os.path.join(DATASET_DIR, 'images/train')}
val: {os.path.join(DATASET_DIR, 'images/val')}
test: {os.path.join(DATASET_DIR, 'images/test')}

names:
"""

for class_id, label in enumerate(unique_labels):
    data_yaml += f"  {class_id}: {label}\n"

with open("data.yaml", "w") as f:
    f.write(data_yaml)

print("Created data.yaml file")

# 6. Bắt đầu training với MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("car_detection_yolo11n")

with mlflow.start_run(run_name="data_preparation"):
    # Log artifacts
    mlflow.log_artifact(os.path.join(DATA_DIR, "image.txt"))
    mlflow.log_artifact(os.path.join(DATA_DIR, "labels.csv"))
    mlflow.log_artifact("data.yaml")
    
# 7. Training model với CPU
print("Starting training with CPU...")

with mlflow.start_run(run_name="model_training_cpu"):
    # Log parameters
    mlflow.log_param("model", "yolov11n")
    mlflow.log_param("epochs", 20)  
    mlflow.log_param("batch_size", 8)  
    mlflow.log_param("image_size", 416)
    mlflow.log_param("device", "cpu")
    
    # Start training
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="data.yaml",
        epochs=20,  
        batch=8,    
        imgsz=416,
        device="cpu",  
        project="runs/train",
        name="exp_cpu"
    )
    
    # Log metrics
    metrics = model.metrics.results_dict
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
    
    # Log artifacts
    mlflow.log_artifacts("runs/train/exp_cpu")
    
    print("Training completed!")

# 8. Đánh giá model
print("Evaluating model...")

with mlflow.start_run(run_name="model_evaluation_cpu"):
    best_model = YOLO("runs/train/exp_cpu/weights/best.pt")
    
    # Evaluate on test set
    val_results = best_model.val(data="data.yaml", device="cpu")
    
    # Log metrics
    for key, value in val_results.results_dict.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
    
    # Run prediction
    results = best_model.predict(
        source=os.path.join(DATASET_DIR, "images/test"),
        save=True,
        project="runs/detect",
        name="predict_cpu",
        device="cpu"
    )
    
    # Log predictions
    mlflow.log_artifacts("runs/detect/predict_cpu")
    
    print("Evaluation completed!")

print("Car detection pipeline with CPU completed successfully!")
