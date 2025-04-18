import streamlit as st
import os
from PIL import Image
from serving_pipeline.utils import get_image_array, get_boxes_for_image, draw_boxes_on_image

st.set_page_config(page_title="Multi-label Object Detection", layout="centered")
st.title("🧠 Visualize Labels from CSV")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_np = get_image_array(uploaded_file)
    st.image(image_np, caption="Uploaded Image", use_column_width=True)

    if st.button("Draw Bounding Boxes"):
        image_id = os.path.splitext(uploaded_file.name)[0]
        h, w = image_np.shape[:2]
        boxes = get_boxes_for_image(image_id, w, h)
        image_with_boxes = draw_boxes_on_image(image_np.copy(), boxes)
        st.image(image_with_boxes, caption="Labeled Bounding Boxes", use_column_width=True)

#streamlit run serving_pipeline/ui.py
