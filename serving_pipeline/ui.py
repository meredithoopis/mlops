import streamlit as st
import os
from PIL import Image
from utils import get_image_array, get_boxes_for_image, draw_boxes_on_image
from ultralytics import YOLO
import cv2

# Mapping labels
YOLO_LABEL_MAP = {
    "car": "Car",
    "truck": "Truck",
    "bus": "Bus",
    "taxi": "Taxi",
    "golf cart": "Golf cart",
    "cart": "Cart",
    "vehicle registration plate": "Vehicle registration plate",
}


st.set_page_config(page_title="Multi-label Object Detection", layout="centered")
st.title("💆🏻 Multi-label Object Detection")
st.info('🚍 Welcome to the app!')

# upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# model
model = YOLO('yolo11n.pt')

# confidence slider
confidence_threshold = st.slider("Select confidence threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded_file is not None:
    image_np = get_image_array(uploaded_file)
    st.image(image_np, caption="Uploaded Image", use_container_width=True)

    if st.button("Draw Bounding Boxes", key="draw_button"):
        image_id = os.path.splitext(uploaded_file.name)[0]
        h, w = image_np.shape[:2]
        boxes = get_boxes_for_image(image_id, w, h)

        if boxes:  # If found in CSV, use CSV labels
            image_with_boxes = draw_boxes_on_image(image_np.copy(), boxes)
            st.image(image_with_boxes, caption="CSV Bounding Boxes", use_container_width=True)
            st.success("CSV-based detection completed successfully!", icon=":material/check:")
        else:  # If not found in CSV, use YOLO to detect
            results = model(image_np)[0]
            filtered_boxes = []

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    filtered_boxes.append(((x1, y1), (x2, y2), label, conf))

            image_with_boxes = draw_boxes_on_image(image_np.copy(), filtered_boxes)
            st.image(image_with_boxes, caption="YOLO Bounding Boxes", use_container_width=True)
            st.success("YOLO detection completed successfully!", icon=":material/check:")


#streamlit run serving_pipeline/ui.py
#streamlit run ui.py