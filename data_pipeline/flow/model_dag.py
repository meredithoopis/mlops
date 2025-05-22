from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os, shutil, random, yaml
import pandas as pd
import cv2, numpy as np
import joblib
from ultralytics import YOLO
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import mlflow
from mlflow.tracking import MlflowClient
import psycopg2
import logging
from dotenv import load_dotenv
load_dotenv() 

#Load environment variables
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT", 5432))

default_args = {
    'owner': 'airflow',
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    
}
dag = DAG(
    dag_id='yolo_train_pipeline',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 5, 6),
    catchup=False,
    description="YOLOv11 training pipeline",
)

#Ensure specific directories exist 
BASE_DIR = "/opt/airflow"
DATA_DIR = f"{BASE_DIR}/data"
LABELS_DIR = f"{BASE_DIR}/labels"
DATASET_DIR = f"{BASE_DIR}/dataset"
LABELS_CSV = f"{BASE_DIR}/vehicle_labels.csv"
MLRUNS_DIR = f"{BASE_DIR}/mlruns"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

class_mapping = {'Car': 0, 'Taxi': 1, 'Truck': 2, 'Bus': 3}

def init_config(**kwargs):
    ti = kwargs['ti']
    tracking_db = f"{MLRUNS_DIR}/mlruns.db"
    os.makedirs(os.path.dirname(tracking_db), exist_ok=True)
    open(tracking_db, 'a').close()
    ti.xcom_push(key='tracking_uri', value=f'sqlite:///{tracking_db}')
    ti.xcom_push(key='experiment_name', value='YOLOv11_Experiments')
    ti.xcom_push(key='model_name', value='YOLOv11_Model')
    ti.xcom_push(key='dataset_dir', value=DATASET_DIR)
    ti.xcom_push(key='data_yaml', value=f"{DATASET_DIR}/data.yaml")
    ti.xcom_push(key='promotion_threshold', value=0.7)

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
    )

def fetch_from_db():
    conn = get_db_connection()
    cur = conn.cursor()
    labels = []
    cur.execute("SELECT image_id, image_data FROM images;")
    for image_id, image_data in cur.fetchall():
        img_path = os.path.join(DATA_DIR, f"{image_id}.jpg")
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(img_path, img)
            cur.execute("SELECT x_min, x_max, y_min, y_max, label_name FROM labels WHERE image_id = %s;", (image_id,))
            for row in cur.fetchall():
                labels.append({
                    'ImageID': image_id,
                    'XMin': row[0], 'XMax': row[1], 'YMin': row[2], 'YMax': row[3], 'LabelName_Text': row[4]
                })
    cur.close(); conn.close()
    pd.DataFrame(labels).to_csv(LABELS_CSV, index=False)

def prepare_labels():
    labels = pd.read_csv(LABELS_CSV).drop_duplicates()
    for image_id in labels['ImageID'].unique():
        with open(os.path.join(LABELS_DIR, f"{image_id}.txt"), 'w') as f:
            for _, row in labels[labels['ImageID'] == image_id].iterrows():
                if row['LabelName_Text'] in class_mapping:
                    class_id = class_mapping[row['LabelName_Text']]
                    x_center = (row['XMin'] + row['XMax']) / 2
                    y_center = (row['YMin'] + row['YMax']) / 2
                    width = row['XMax'] - row['XMin']
                    height = row['YMax'] - row['YMin']
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def split_dataset():
    images = [img for img in os.listdir(DATA_DIR) if img.endswith(".jpg") and os.path.exists(os.path.join(LABELS_DIR, img.replace(".jpg", ".txt")))]
    random.seed(42)
    train_imgs = random.sample(images, int(0.9 * len(images)))
    val_imgs = [img for img in images if img not in train_imgs]
    for split in ['train', 'val']:
        os.makedirs(f"{DATASET_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/labels/{split}", exist_ok=True)
    for img in train_imgs:
        shutil.copy(f"{DATA_DIR}/{img}", f"{DATASET_DIR}/images/train/{img}")
        shutil.copy(f"{LABELS_DIR}/{img.replace('.jpg','.txt')}", f"{DATASET_DIR}/labels/train/{img.replace('.jpg','.txt')}")
    for img in val_imgs:
        shutil.copy(f"{DATA_DIR}/{img}", f"{DATASET_DIR}/images/val/{img}")
        shutil.copy(f"{LABELS_DIR}/{img.replace('.jpg','.txt')}", f"{DATASET_DIR}/labels/val/{img.replace('.jpg','.txt')}")

def create_data_yaml(**kwargs):
    dataset_dir = kwargs['ti'].xcom_pull(key='dataset_dir', task_ids='init_config')
    yaml_path = f"{dataset_dir}/data.yaml"
    data_yaml = f"""train: {dataset_dir}/images/train
val: {dataset_dir}/images/val
nc: 4
names: ['Car', 'Taxi', 'Truck', 'Bus']
# Data augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 15.0
translate: 0.1
scale: 0.5
shear: 0.0
flipud: 0.5
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
"""
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    kwargs['ti'].xcom_push(key='data_yaml', value=yaml_path)
    
    
def train_yolo_full(**kwargs):
    if not os.path.exists("yolo11n.pt"):
        os.system("curl -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt")

    dataset_dir = kwargs['ti'].xcom_pull(key='dataset_dir', task_ids='init_config')
    data_yaml = kwargs['ti'].xcom_pull(key='data_yaml', task_ids='init_config')

    model = YOLO("yolo11n.pt")
    model.train(
        data=data_yaml,
        epochs=1,
        imgsz=640,
        batch=16,
        device="cpu",
        optimizer="AdamW",
        lr0=0.0005,
        name="train_hyt"
    )

#This is for testing purposes only 
def train_yolo_subset(**kwargs):
    from pathlib import Path
    if not os.path.exists("yolo11n.pt"):
        os.system("curl -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt")
    dataset_dir = kwargs['ti'].xcom_pull(key='dataset_dir', task_ids='init_config')
    subset_dir = f"{dataset_dir}/subset"
    subset_img = f"{subset_dir}/images/train"
    subset_lbl = f"{subset_dir}/labels/train"
    os.makedirs(subset_img, exist_ok=True)
    os.makedirs(subset_lbl, exist_ok=True)
    train_img_paths = list(Path(f"{dataset_dir}/images/train").glob("*.jpg"))
    sampled = random.sample(train_img_paths, min(50, len(train_img_paths)))
    for img_path in sampled:
        label_path = Path(f"{dataset_dir}/labels/train") / img_path.with_suffix('.txt').name
        shutil.copy(img_path, subset_img)
        shutil.copy(label_path, subset_lbl)
    subset_yaml = f"{subset_dir}/data.yaml"
    data_dict = {
        "train": subset_img,
        "val": f"{dataset_dir}/images/val",
        "nc": 4,
        "names": ['Car', 'Taxi', 'Truck', 'Bus']
    }
    with open(subset_yaml, 'w') as f:
        yaml.dump(data_dict, f)
    model = YOLO("yolo11n.pt")
    model.train(data=subset_yaml, epochs=1, imgsz=416, batch=16, device="cpu", optimizer="AdamW", lr0=0.0005, name="train_hyt")

def log_to_mlflow(**kwargs):
    log = logging.getLogger("airflow.task")
    try:
        ti = kwargs['ti']
        uri = ti.xcom_pull(task_ids='init_config', key='tracking_uri')
        experiment = ti.xcom_pull(task_ids='init_config', key='experiment_name')
        model_name = ti.xcom_pull(task_ids='init_config', key='model_name')
        threshold = ti.xcom_pull(task_ids='init_config', key='promotion_threshold')

        tracking_uri = MLRUNS_DIR  
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        results_path = "runs/detect/train_hyt/results.csv"
        model_path = "runs/detect/train_hyt/weights/best.pt"
        train_dir = "runs/detect/train_hyt"
        results = pd.read_csv(results_path)
        metrics = {
            "mAP50": results["metrics/mAP50(B)"].iloc[-1],
            "mAP50-95": results["metrics/mAP50-95(B)"].iloc[-1],
            "train_box_loss": results["train/box_loss"].iloc[-1],
            "val_box_loss": results["val/box_loss"].iloc[-1]
        }

        with mlflow.start_run(run_name="YOLOv11_Training") as run:
            mlflow.log_params({"epochs": 1, "imgsz": 416, "batch": 16, "optimizer": "AdamW", "lr0": 0.0005})
            mlflow.log_artifact(results_path)
            mlflow.log_artifact(model_path)
            plot_files = [
                "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                "F1_curve.png", "P_curve.png", "R_curve.png", "PR_curve.png",
                "labels.jpg", "labels_correlogram.jpg"
            ]
            for f in plot_files:
                plot_path = os.path.join(train_dir, f)
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, artifact_path="training_plots")

            for f in os.listdir(train_dir):
                if f.startswith("val_batch") and f.endswith(".jpg"):
                    mlflow.log_artifact(os.path.join(train_dir, f), artifact_path="val_examples")
            
            mlflow.log_metrics(metrics)            
            signature = ModelSignature(
                inputs=Schema([TensorSpec(np.dtype("float32"), (-1, 3, 640, 640))]),
                outputs=Schema([TensorSpec(np.dtype("float32"), (-1, -1))])
)
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri, model_name)
            if metrics["mAP50"] >= threshold:
                MlflowClient().transition_model_version_stage(name=model_name, version=result.version, stage="Production")
    except Exception as e:
        import traceback
        log.error("❌ log_to_mlflow failed: %s", str(e))
        log.error(traceback.format_exc())
        raise


init_task = PythonOperator(task_id="init_config", python_callable=init_config, provide_context=True, dag=dag)
fetch_task = PythonOperator(task_id="fetch_from_db", python_callable=fetch_from_db, dag=dag)
prepare_task = PythonOperator(task_id="prepare_labels", python_callable=prepare_labels, dag=dag)
split_task = PythonOperator(task_id="split_dataset", python_callable=split_dataset, dag=dag)
yaml_task = PythonOperator(task_id="create_data_yaml", python_callable=create_data_yaml, provide_context=True, dag=dag)
train_task = PythonOperator(task_id="train_yolo_full", python_callable=train_yolo_full, provide_context=True, dag=dag)
mlflow_task = PythonOperator(task_id="log_to_mlflow", python_callable=log_to_mlflow, provide_context=True, dag=dag)

init_task >> fetch_task >> prepare_task >> split_task >> yaml_task >> train_task >> mlflow_task

#Acess the MLflow UI: mlflow ui --backend-store-uri ./mlruns --port 5000 (In the same folder)