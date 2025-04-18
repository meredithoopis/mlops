import pandas as pd
import cv2
import os
from PIL import Image
import numpy as np
from config import CSV_PATH

df = pd.read_csv(CSV_PATH)

def get_image_array(image_file):
    return np.array(Image.open(image_file).convert("RGB"))

def get_boxes_for_image(image_id, image_width, image_height):
    boxes = []
    rows = df[df["ImageID"] == image_id]

    for _, row in rows.iterrows():
        x1 = int(row["XMin"] * image_width)
        y1 = int(row["YMin"] * image_height)
        x2 = int(row["XMax"] * image_width)
        y2 = int(row["YMax"] * image_height)
        label = row["LabelName_Text"]
        boxes.append(((x1, y1), (x2, y2), label))
    return boxes

def draw_boxes_on_image(image_np, boxes):
    for (x1, y1), (x2, y2), label in boxes:
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image_np
