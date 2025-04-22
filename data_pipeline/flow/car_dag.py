from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import cv2
import psycopg2
import numpy as np
import pandas as pd
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
IMAGES_DIR = "/opt/airflow/images"   # Folder containing .jpg files
LABELS_CSV = "/opt/airflow/labels.csv"  # Single CSV with all labels

#db_host = 'localhost'
DB_HOST     = "postgres"
DB_NAME     = "airflow"
DB_USER     = "airflow"
DB_PASSWORD = "airflow"

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
LABELS_DF = pd.read_csv(LABELS_CSV)

# --- Task functions ---
def load_image(image_name, **kwargs):
    """Read and encode image as base64 for downstream tasks."""
    path = os.path.join(IMAGES_DIR, image_name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    _, buffer = cv2.imencode('.jpg', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    
    image_id = os.path.splitext(image_name)[0] #added 
    ti = kwargs['ti']
    ti.xcom_push(key='image_name', value=image_name)
    ti.xcom_push(key='image_id', value=image_id)
    ti.xcom_push(key='image_b64', value=b64_str)
    logging.info(f"Loaded and encoded image {image_name}")

def preprocess_image(**kwargs):
    """Decode base64, resize, re-encode, and push for saving."""
    ti = kwargs['ti']
    image_name = ti.xcom_pull(key='image_name', task_ids=kwargs['task_instance'].task_id.replace('preprocess', 'load'))
    b64_str = ti.xcom_pull(key='image_b64', task_ids=f'load_{image_name}')
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224))
    _, buffer = cv2.imencode('.jpg', img_resized)
    proc_b64 = base64.b64encode(buffer).decode('utf-8')
    ti.xcom_push(key='proc_b64', value=proc_b64)
    logging.info(f"Preprocessed image {image_name}")
    
    
def save_image_to_db(**kwargs):
    """Insert or update image binary into Postgres."""
    ti = kwargs['ti']
    image_name = ti.xcom_pull(key='image_name', task_ids=kwargs['task_instance'].task_id.replace('save', 'load'))
    image_id = ti.xcom_pull(key='image_id', task_ids=kwargs['task_instance'].task_id.replace('save', 'load'))
    proc_b64 = ti.xcom_pull(key='proc_b64', task_ids=f'preprocess_{image_name}')
    img_bytes = base64.b64decode(proc_b64)

    conn = get_db_connection()
    cur = conn.cursor()
    # Create table if missing
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            image_data BYTEA NOT NULL
        )""")
    image_id = os.path.splitext(image_name)[0]
    cur.execute(
        "INSERT INTO images (image_id, image_data) VALUES (%s, %s)"
        " ON CONFLICT (image_id) DO UPDATE SET image_data = EXCLUDED.image_data",
        (image_id, psycopg2.Binary(img_bytes))
    )
    conn.commit()
    cur.close()
    conn.close()
    logging.info(f"Saved image {image_id} to DB")
    
    
def load_labels_for_image(image_name, **kwargs):
    """Pull all label rows for a given image ID from preloaded DataFrame."""
    image_id = os.path.splitext(image_name)[0]
    image_id = image_id.replace('_jpg','')
    labels_df = pd.read_csv(LABELS_CSV)
    records = labels_df[labels_df['ImageID'] == image_id]
    #records = LABELS_DF[LABELS_DF['ImageID'] == image_id]
    recs = records[['XMin', 'XMax', 'YMin', 'YMax', 'LabelName_Text']].values.tolist()
    logging.info(f"Found {len(recs)} labels for image {image_id}")
    return recs  # XCom return used downstream


def save_labels_to_db(image_name, **kwargs):
    """Iterate over all label records and insert into Postgres labels table."""
    ti = kwargs['ti']
    image_id = os.path.splitext(image_name)[0]
    image_id = image_id.replace('_jpg','')
    recs = ti.xcom_pull(task_ids=f'load_labels_{image_name}', key='return_value') or []

    if not recs:
        logging.warning(f"No labels to insert for image {image_id}")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    # Create table if missing
    cur.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id SERIAL PRIMARY KEY,
            image_id TEXT REFERENCES images(image_id) ON DELETE CASCADE,
            x_min REAL, x_max REAL, y_min REAL, y_max REAL,
            label_name TEXT
        )""")

    for xmin, xmax, ymin, ymax, label_name in recs:
        cur.execute(
            "INSERT INTO labels (image_id, x_min, x_max, y_min, y_max, label_name)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            (image_id, xmin, xmax, ymin, ymax, label_name)
        )
    conn.commit()
    cur.close()
    conn.close()
    logging.info(f"Inserted {len(recs)} labels for image {image_id}")
    
    

# --- DAG definition ---
with DAG(
    dag_id='car_pipeline',
    start_date=datetime(2025, 4, 10),
    schedule_interval=None,
    catchup=False,
    default_args={'owner': 'airflow'}
) as dag:
    # Discover all images in the dataset
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]

    for image_name in image_files:
        load = PythonOperator(
            task_id=f'load_{image_name}',
            python_callable=load_image,
            op_args=[image_name]
        )

        preprocess = PythonOperator(
            task_id=f'preprocess_{image_name}',
            python_callable=preprocess_image
        )

        save_img = PythonOperator(
            task_id=f'save_{image_name}',
            python_callable=save_image_to_db
        )

        load_lbl = PythonOperator(
            task_id=f'load_labels_{image_name}',
            python_callable=load_labels_for_image,
            op_args=[image_name]
        )

        save_lbl = PythonOperator(
            task_id=f'save_labels_{image_name}',
            python_callable=save_labels_to_db,
            op_args=[image_name]
        )

        # Define task dependencies
        load >> preprocess >> save_img
        load_lbl >> save_lbl