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
import math

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
IMAGES_DIR = "/opt/airflow/data"
LABELS_CSV = "/opt/airflow/vehicle_labels.csv"
DB_HOST = "postgres"
DB_NAME = "airflow"
DB_USER = "airflow"
DB_PASSWORD = "airflow"
DB_PORT = 5432
BATCH_SIZE = 100

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

LABELS_DF = pd.read_csv(LABELS_CSV)

def init_schema():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            image_data BYTEA NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id BIGSERIAL PRIMARY KEY,
            image_id TEXT REFERENCES images(image_id) ON DELETE CASCADE,
            x_min REAL, x_max REAL, y_min REAL, y_max REAL,
            label_name TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    logging.info("Schema initialized.")

def process_batch_images(batch_images, **kwargs):
    conn = get_db_connection()
    cur = conn.cursor()

    for image_name in batch_images:
        try:
            image_path = os.path.join(IMAGES_DIR, image_name)
            img = cv2.imread(image_path)
            if img is None:
                logging.warning(f"Image not found or unreadable: {image_path}")
                continue

            img_bytes_resized = cv2.imencode('.jpg', cv2.resize(img, (224, 224)))[1].tobytes()
            image_id = os.path.splitext(image_name)[0]

            # Insert or update image
            cur.execute(
                "INSERT INTO images (image_id, image_data) VALUES (%s, %s) "
                "ON CONFLICT (image_id) DO UPDATE SET image_data = EXCLUDED.image_data",
                (image_id, psycopg2.Binary(img_bytes_resized))
            )

            # Get all labels for this image
            records = LABELS_DF[LABELS_DF['ImageID'] == image_id]
            recs = records[['XMin', 'XMax', 'YMin', 'YMax', 'LabelName_Text']].values.tolist()
            logging.info(f"Found {len(recs)} labels for image {image_id}")

            for xmin, xmax, ymin, ymax, label_name in recs:
                cur.execute(
                    "INSERT INTO labels (image_id, x_min, x_max, y_min, y_max, label_name) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (image_id, xmin, xmax, ymin, ymax, label_name)
                )

            # Commit after each image and its labels are processed
            conn.commit()

        except Exception as e:
            logging.error(f"Processing failed for image {image_name}: {e}")
            conn.rollback()

    cur.close()
    conn.close()
    logging.info("Finished processing current batch of images.")


with DAG(
    dag_id='car_pipeline_batch',
    start_date=datetime(2025, 4, 10),
    schedule_interval=None,
    catchup=False,
    default_args={'owner': 'airflow'}
) as dag:

    init_task = PythonOperator(
        task_id='init_schema',
        python_callable=init_schema
    )

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]
    num_batches = math.ceil(len(image_files) / BATCH_SIZE)

    for i in range(num_batches):
        batch_files = image_files[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_task = PythonOperator(
            task_id=f'process_batch_{i+1}',
            python_callable=process_batch_images,
            op_args=[batch_files]
        )
        init_task >> batch_task
