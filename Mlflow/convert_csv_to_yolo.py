import pandas as pd
import os

# Mapping từ nhãn sang class id
label_map = {
    "Car": 0,
    "Taxi": 1,
    "Truck": 2,
    "Bus": 3,
    "Cart": 4,
    "Golf cart": 5,
}

# Đường dẫn đến file CSV
csv_file = "labels/car_labels.csv"

# Folder để lưu các file .txt
output_dir = "labels/train"
os.makedirs(output_dir, exist_ok=True)

# Đọc CSV
df = pd.read_csv(csv_file)

# Nhóm theo ảnh
for image_id, group in df.groupby("ImageID"):
    lines = []
    for _, row in group.iterrows():
        class_id = label_map.get(row["LabelName_Text"])
        if class_id is None:
            continue  # bỏ qua nếu nhãn không hợp lệ

        x_center = (row["XMin"] + row["XMax"]) / 2
        y_center = (row["YMin"] + row["YMax"]) / 2
        width = row["XMax"] - row["XMin"]
        height = row["YMax"] - row["YMin"]

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)

    # Ghi ra file .txt riêng cho ảnh
    out_file = os.path.join(output_dir, f"{image_id}.txt")
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
