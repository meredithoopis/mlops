import pandas as pd
import os
import random
import shutil
import glob

data_dir = "data"
labels_csv = "vehicle_labels.csv"
labels_dir = "/Users/admin/Car-detection-serving-model/open_images_project/labels"
dataset_dir = "/Users/admin/Car-detection-serving-model/open_images_project/dataset"
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
    if os.path.exists(lbl) and os.path.getsize(lbl) > 0:  # Chỉ lấy ảnh có nhãn
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

class_names = list(class_mapping.keys())
print("Phân bố lớp trong dataset lớn sau khi làm sạch:")
print("Train:")
for i in range(len(class_names)):
    print(f"{class_names[i]}: {class_counts[i]} instances")
print(f"Số ảnh train: {len(train_imgs)}")
print(f"Số ảnh val: {len(val_imgs)}")

if invalid_files:
    print("\nCác file chứa class_id không hợp lệ:")
    for file, class_id in invalid_files:
        print(f"File: {file}, class_id: {class_id}")
else:
    print("\nKhông tìm thấy class_id không hợp lệ.")


## Tạo data.yaml
import os

dataset_dir = "/Users/admin/Car-detection-serving-model/open_images_project/dataset"
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
print("Đã tạo data.yaml với 4 lớp và augmentation")



from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(
    data="/Users/admin/Car-detection-serving-model/open_images_project/dataset/data.yaml",
    epochs=60,
    imgsz=640,  
    batch=16,
    device="cpu",
    optimizer="AdamW",
    lr0=0.0005, 
    name="train_large_new_final"
)

