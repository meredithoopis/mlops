1. Chuẩn bị dữ liệu
import pandas as pd
import os
import random
import shutil
import glob

data_dir = "data"
labels_csv = "vehicle_labels.csv"
labels_dir = "open_images_project/labels"
dataset_dir = "open_images_project/dataset"
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)

class_mapping = {
    'Car': 0,
    'Taxi': 1,
    'Truck': 2,
    'Bus': 3
}

labels = pd.read_csv(labels_csv)
labels = labels.drop_duplicates(subset=['ImageID', 'LabelName_Text', 'XMin', 'YMin', 'XMax', 'YMax'])

for image_id in labels['ImageID'].unique():
    image_labels = labels[labels['ImageID'] == image_id]
    output_file = os.path.join(labels_dir, f"{image_id}.txt")
    with open(output_file, 'w') as f:
        for _, row in image_labels.iterrows():
            class_name = row['LabelName_Text']
            if class_name in class_mapping:
                class_id = class_mapping[class_name]
                x_min = row['XMin']
                x_max = row['XMax']
                y_min = row['YMin']
                y_max = row['YMax']
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

images_with_labels = []
for img in os.listdir(data_dir):
    if not img.endswith(".jpg"):
        continue
    lbl = os.path.join(labels_dir, img.replace(".jpg", ".txt"))
    if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
        images_with_labels.append(img)

random.seed(1610)
train_imgs = random.sample(images_with_labels, int(0.9 * len(images_with_labels)))
val_imgs = [img for img in images_with_labels if img not in train_imgs]

for img in train_imgs:
    shutil.copy(os.path.join(data_dir, img), f"{dataset_dir}/images/train/{img}")
    lbl = img.replace(".jpg", ".txt")
    shutil.copy(os.path.join(labels_dir, lbl), f"{dataset_dir}/labels/train/{lbl}")

for img in val_imgs:
    shutil.copy(os.path.join(data_dir, img), f"{dataset_dir}/images/val/{img}")
    lbl = img.replace(".jpg", ".txt")
    shutil.copy(os.path.join(labels_dir, lbl), f"{dataset_dir}/labels/val/{lbl}")

class_counts = {i: 0 for i in range(len(class_mapping))}
invalid_files = []
for label_file in glob.glob(f"{dataset_dir}/labels/*/*.txt"):
    with open(label_file, "r") as f:
        for line in f:
            if line.strip():
                class_id = int(line.split()[0])
                if class_id in class_counts:
                    class_counts[class_id] += 1
                else:
                    invalid_files.append((label_file, class_id))

                    
2. Tạo file data.yaml
import os

dataset_dir = "open_images_project/dataset"
data_yaml_content = f"""train: {dataset_dir}/images/train
val: {dataset_dir}/images/val
nc: 4
names: ['Car', 'Taxi', 'Truck', 'Bus']

hsv_h: 0.015  # Hue
hsv_s: 0.7    # Saturation
hsv_v: 0.4    # Value
degrees: 15.0  # Rotation
translate: 0.1  # Translation
scale: 0.5     # Scaling
shear: 0.0     # Shear
flipud: 0.5    # Flip up-down
fliplr: 0.5    # Flip left-right
mosaic: 1.0    # Mosaic
mixup: 0.0     # Mixup
"""
with open(f"{dataset_dir}/data.yaml", "w") as f:
    f.write(data_yaml_content)



Mô hình được huấn luyện qua 4 giai đoạn (tổng cộng 60 epochs):
1. 30 epochs (`train_large_new`).
2. 10 epochs (`train_large_new_extended`).
3. 10 epochs (`train_large_new_extended_2`).
4. 10 epochs (`train_large_new_final`).

Đoạn code dưới đây là phiên bản gộp tương đương (60 epochs), nhưng thực tế mình đã chạy tách rời:
3. Huấn luyện mô hình
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(
    data="open_images_project/dataset/data.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    device="cpu",
    optimizer="AdamW",
    lr0=0.0005,
    name="train_large_new_final"
)
4. Log metrics, tham số và artifacts vào MLflow
import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("YOLOv11_Experiments")

results_df = pd.read_csv("runs/detect/train_large_new_final/results.csv")

with mlflow.start_run(run_name="YOLOv11_Training_Final"):
    # Log metrics
    metrics = {
        "mAP50": results_df["metrics/mAP50(B)"].iloc[-1],
        "mAP50-95": results_df["metrics/mAP50-95(B)"].iloc[-1],
        "train_box_loss": results_df["train/box_loss"].iloc[-1],
        "val_box_loss": results_df["val/box_loss"].iloc[-1]
    }
    mlflow.log_metrics(metrics)

    # Log tham số
    params = {
        "epochs": 60,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 0.0005
    }
    mlflow.log_params(params)

    # Log artifacts
    mlflow.log_artifact("open_images_project/dataset/data.yaml")
    mlflow.log_artifact("runs/detect/train_large_new_final/weights/best.pt")
    mlflow.log_artifact("runs/detect/train_large_new_final/results.png", "training_plots")

5.Đăng ký mô hình vào MLflow Model Registry
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("YOLOv11_Experiments")

with mlflow.start_run(run_name="YOLOv11_Training_Final"):
    model_path = "runs/detect/train_large_new_final/weights/best.pt"
    mlflow.log_artifact(model_path)

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    result = mlflow.register_model(model_uri, "YOLOv11_Model")

    client = MlflowClient()
    tags = {
        "model_type": "YOLOv11",
        "task": "vehicle_detection",
        "imgsz": "640",
        "epochs": "60",
        "optimizer": "AdamW"
    }
    for key, value in tags.items():
        client.set_model_version_tag("YOLOv11_Model", result.version, key, value)

    client.set_registered_model_tag("YOLOv11_Model", "description", "YOLOv11 model for vehicle detection")

    mAP50 = results_df["metrics/mAP50(B)"].iloc[-1]
    if mAP50 >= 0.7:
        client.transition_model_version_stage(
            name="YOLOv11_Model",
            version=result.version,
            stage="Production"
        )
        print(f"Model version {result.version} transitioned to Production stage")
 

                    
