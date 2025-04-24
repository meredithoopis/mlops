import streamlit as st
import os
from PIL import Image
import cv2
from ultralytics import YOLO

# Đặt cấu hình trang Streamlit
st.set_page_config(page_title="Multi-label Object Detection", layout="centered")

from utils import get_image_array, get_boxes_for_image, draw_boxes_on_image
from downloader import download_resources

st.title("🛸 Multi-label Object Detection")
st.info('🚍 Welcome to the app!')

# Tải tài nguyên cần thiết (model, v.v.)
resources_ready = download_resources()

if resources_ready:
    # Load model YOLO
    model = YOLO('yolo11n.pt')

    available_objects = ["car", "truck", "bus", "taxi", "vehicle registration plate"]

    # Upload ảnh
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Chọn các object muốn detect
    selected_objects = st.multiselect(
        "Select objects to detect",
        options=available_objects,
        default=[]
    )

    # Nếu không chọn gì, detect tất cả object mà YOLO biết
    detect_all = len(selected_objects) == 0

    # Chuyển các object đã chọn thành dạng lowercase để dễ so sánh
    selected_objects_lower = [obj.lower() for obj in selected_objects]

    # Confidence threshold
    confidence_threshold = st.slider("Select confidence threshold", 0.0, 1.0, 0.5, 0.01)

    if uploaded_file is not None:
        image_np = get_image_array(uploaded_file)
        st.image(image_np, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect objects", key="detect_button", icon=":material/local_car_wash:"):
            image_id = os.path.splitext(uploaded_file.name)[0]
            h, w = image_np.shape[:2]
            boxes = get_boxes_for_image(image_id, w, h)

            if boxes:
                # Nếu ảnh có trong file CSV (bounding box có sẵn)
                image_with_boxes = draw_boxes_on_image(image_np.copy(), boxes)
                st.image(image_with_boxes, caption="CSV Bounding Boxes", use_container_width=True)
                st.success("CSV-based detection completed successfully!", icon=":material/check:")
            else:
                # Dùng YOLO để detect
                results = model(image_np)[0]
                filtered_boxes = []

                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        label_lower = label.lower()

                        if detect_all or label_lower in selected_objects_lower:
                            filtered_boxes.append(((x1, y1), (x2, y2), label, conf))

                image_with_boxes = draw_boxes_on_image(image_np.copy(), filtered_boxes)
                st.image(image_with_boxes, caption="YOLO Bounding Boxes", use_container_width=True)
                st.success("YOLO detection completed successfully!", icon=":material/check:")

else:
    st.error("❌ Failed to download required resources. Please check the internet connection and try again.")


#streamlit run serving_pipeline/ui.py
#streamlit run ui.py