import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # serving_pipeline/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) 

CSV_PATH = os.path.join(BASE_DIR, "car_labels.csv")
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data_pipeline", "traindata", "data")
