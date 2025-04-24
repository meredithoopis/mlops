import os
import gdown
import streamlit as st

def download_file(file_id, output_path):
    """Download file from Google Drive using file ID"""
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {os.path.basename(output_path)}..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            st.success(f"Downloaded {os.path.basename(output_path)} successfully!")
    else:
        st.write(f"---- {os.path.basename(output_path)} is already ----")

@st.cache_resource
def download_resources():
    """Download all needed resources"""
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    
    # Replace these with your actual file IDs from Google Drive
    model_file_id = "1pEM9YpONm_8aBqVhAT3G_iS1BodVEP5v"  # ID yolo11n.pt
    csv_file_id = "17Per3L0smmP-I0EN1SUI8zG_jWAgIjhp"      # ID car_labels.csv
    
    # Download model
    download_file(model_file_id, "yolo11n.pt")
    
    # Download CSV
    download_file(csv_file_id, "car_labels.csv")
    
    return True

# car_labels
# https://drive.google.com/file/d/17Per3L0smmP-I0EN1SUI8zG_jWAgIjhp/view?usp=sharing

# yolo11n.pt
# https://drive.google.com/file/d/1pEM9YpONm_8aBqVhAT3G_iS1BodVEP5v/view?usp=sharing