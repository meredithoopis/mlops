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
        probability = row["Confidence"]
        boxes.append(((x1, y1), (x2, y2), label, probability))
    return boxes

# RGB color
COLOR_MAP = {
    "Car": (0, 128, 255),                            # Blue
    "car": (0, 128, 255),                            # Blue
    "Vehicle registration plate": (0, 204, 102),     # Green
    "vehicle registration plate": (0, 204, 102),     # Green
    "Truck": (255, 102, 178),                        # Pink
    "truck": (255, 102, 178),                        # Pink
    "Taxi": (0, 0, 255),                             # Red
    "taxi": (0, 0, 255),                             # Red
    "Bus": (255, 0, 255),                            # Magenta
    "bus": (255, 0, 255),                            # Magenta
    "Cart": (255, 255, 0),                           # Cyan
    "cart": (255, 255, 0),                           # Cyan
    "Golf cart": (255, 128, 0),                      # Orange
    "golf cart": (255, 128, 0),                      # Orange
}

def draw_boxes_on_image(image_np, boxes):
    for (x1, y1), (x2, y2), label, probability in boxes:
        color = COLOR_MAP.get(label, (192, 192, 192))  # Default Grey if not in map

        # Draw bounding box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

        # Get text size
        label_text = f"{label} {probability:.2f}"  # Format the label with the probability (2 decimal places)
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        # Draw filled rectangle for label background
        cv2.rectangle(image_np, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)

        # Put label text with probability in white
        cv2.putText(image_np, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Larger, bolder text

    return image_np