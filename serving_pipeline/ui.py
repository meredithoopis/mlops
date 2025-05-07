import streamlit as st
import os
from PIL import Image
import cv2
from ultralytics import YOLO  
from db import save_full_feedback, create_table 
import psycopg2
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

try:
    create_table()
except Exception as e:
    st.error(f"❌ Failed to ensure feedback table exists: {e}")
st.set_page_config(page_title="Multi-label Object Detection", layout="centered")

from utils import get_image_array, get_boxes_for_image, draw_boxes_on_image
from downloader import download_resources

st.title("🛸 Multi-label Object Detection")
st.info('🚍 Welcome to the app!')

resources_ready = True 

if resources_ready:
    model_path = os.path.join("..", "tracking_pipeline", "yolo11n.pt") 
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
    else:
        model = YOLO(model_path)

    available_objects = ["car", "truck", "bus", "taxi", "vehicle registration plate"]

    # Upload images 
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Chọn objects to detect
    selected_objects = st.multiselect(
        "Select objects to detect",
        options=available_objects,
        default=[]
    )

    # Default is detect all  
    detect_all = len(selected_objects) == 0
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
                image_with_boxes = draw_boxes_on_image(image_np.copy(), boxes)
                st.image(image_with_boxes, caption="CSV Bounding Boxes", use_container_width=True)
                st.success("CSV-based detection completed successfully!", icon=":material/check:")
            else:
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
                            
                             # 💾 Save each detection into database
                            try:
                                save_full_feedback(
                                    image_id=os.path.splitext(uploaded_file.name)[0],
                                    source="user_upload",
                                    label_name=label,
                                    confidence=conf,
                                    x_min=x1 / w,
                                    x_max=x2 / w,
                                    y_min=y1 / h,
                                    y_max=y2 / h,
                                    is_occluded=False,
                                    is_truncated=False,
                                    is_group_of=False,
                                    is_depiction=False,
                                    is_inside=False,
                                    xclick1x=x1 / w,
                                    xclick2x=x2 / w,
                                    xclick3x=x1 / w,
                                    xclick4x=x2 / w,
                                    xclick1y=y1 / h,
                                    xclick2y=y1 / h,
                                    xclick3y=y2 / h,
                                    xclick4y=y2 / h,
                                    labelname_text=label
                                )
                            except Exception as e:
                                 st.error(f"❌ Failed to save feedback: {e}")


                image_with_boxes = draw_boxes_on_image(image_np.copy(), filtered_boxes)
                st.image(image_with_boxes, caption="YOLO Bounding Boxes", use_container_width=True)
                st.success("YOLO detection completed successfully!", icon=":material/check:")

else:
    st.error("❌ Failed to download required resources. Please try again.")


