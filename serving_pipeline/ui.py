import streamlit as st
import os
from PIL import Image
import pandas as pd 
import cv2
from ultralytics import YOLO  
from db import save_full_feedback, create_table 
import psycopg2
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

def find_best_model(mruns_path: str) -> str:
    """
    Find the best model from MLFLOW 
    """
    best_score = -1
    best_model_path = None

    for experiment_id in os.listdir(mruns_path):
        if not experiment_id.isdigit():
            continue

        experiment_path = os.path.join(mruns_path, experiment_id)
        if not os.path.isdir(experiment_path):
            continue

        for run_id in os.listdir(experiment_path):
            run_path = os.path.join(experiment_path, run_id)
            metrics_file = os.path.join(run_path, "metrics", "mAP50")

            if not os.path.isfile(metrics_file):
                continue

            try:
                with open(metrics_file) as f:
                    # Extract the 2nd column (mAP50 value)
                    values = [float(line.strip().split()[1]) for line in f if line.strip()]
                    if not values:
                        continue
                    latest_score = values[-1]
            except Exception as e:
                print(f"Failed to read mAP50 from {metrics_file}: {e}")
                continue

            artifact_model_path = os.path.join(run_path, "artifacts", "best.pt")
            if latest_score > best_score and os.path.exists(artifact_model_path):
                best_score = latest_score
                best_model_path = artifact_model_path 
    return best_model_path


def load_best_or_default_model(): 
    model_choice = st.radio(
        "Choose the model source:",
        ("Use MLflow-trained model", "Use default YOLOv11 model"),
        index=0
    )

    # Paths
    mlruns_dir = os.path.abspath(os.path.join("..", "data_pipeline", "flow", "mlruns"))
    fallback_model_path = os.path.abspath(os.path.join("..", "tracking_pipeline", "yolo11n.pt"))

    if model_choice == "Use MLflow-trained model":
        best_model_path = find_best_model(mlruns_dir)
        if best_model_path and os.path.exists(best_model_path):
            st.success(f"✅ Loaded best model from MLflow")
            return YOLO(fallback_model_path)
        else:
            st.warning("⚠️ No valid MLflow model found. Falling back to YOLOv11.")

    # If user chose default, or fallback is triggered
    if os.path.exists(fallback_model_path):
        st.success(f"✅ Loaded fallback model")
        return YOLO(fallback_model_path)
    else:
        st.error("❌ No model found at all.")
        return None


try:
    create_table()
except Exception as e:
    st.error(f"❌ Failed to ensure feedback table exists: {e}")
st.set_page_config(page_title="Multi-label Object Detection", layout="centered")

from utils import get_image_array, get_boxes_for_image, draw_boxes_on_image
from downloader import download_resources

st.title("🛸 Multi-label Object Detection")
st.info('🚍 Welcome to the app!')

model = load_best_or_default_model()
resources_ready = model is not None
    
if resources_ready:
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


