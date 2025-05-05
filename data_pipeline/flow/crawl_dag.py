from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import cv2
import os
import psycopg2
import requests
from serpapi import GoogleSearch
from dotenv import load_dotenv
from tqdm import tqdm
import json
import subprocess
from PIL import Image, UnidentifiedImageError
import time
from pathlib import Path
import logging 

# Configure logging
logging.basicConfig(level=logging.INFO)

DB_HOST = "172.20.219.28"
DB_NAME = "carrrr"
DB_USER = "airflow"
DB_PASSWORD = "airflow"
DB_PORT = 5432

# Load .env
#API_KEY = "fef2ffbf8d56b87b1920a9fc91809ab0bbc936101777e9863df3d54451f749ee"
dotenv_path = Path('./.env')
load_dotenv(dotenv_path)
API_KEY = os.getenv("API_KEY")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    
def init_schema():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images_crawled (
            image_id TEXT PRIMARY KEY,
            image_data BYTEA NOT NULL
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    logging.info("Schema initialized.")

# Function to download images from Google using SerpAPI
def download_images_from_google(query, output_dir):
    #API_KEY = os.getenv('API_KEY')

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    params = {
        "api_key": API_KEY,
        "engine": "google_images",
        "google_domain": "google.com",
        "q": query,
        "hl": "en",
        "gl": "us",
        "num": 50
    }

    # Perform the search via SerpAPI
    search = GoogleSearch(params)
    results = search.get_dict()

    # Saving results in a JSON file for reference
    with open(f"{output_dir}/res.json", "w") as f:
        json.dump(results, f, indent=4)

    # Download images from the search results
    images_link = [item['original'] for item in results.get("images_results", [])]
    if not images_link:
        print("❌ No image links found.")
        return
    print(f"📸 Found {len(images_link)} image links. Downloading..")

    limit = 50 
    downloaded = 0 
    for i, url in enumerate(tqdm(images_link, desc="Downloading images")):
        if downloaded >= limit:
            break
        filename = f"image_{downloaded + 1}.png"
        save_path = os.path.join(output_dir, filename)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"✅ Image saved to {save_path}")
        except requests.RequestException as e:
            print(f"❌ Failed to download {url}: {e}")
        downloaded += 1 
        time.sleep(0.2)
    #return len(images_link)


def download_videos_from_youtube(query, output_dir):

    # Currently I'm limiting the search to 2 videos
    MAX_FILESIZE_MB = 500
    MAX_FILESIZE = f"{MAX_FILESIZE_MB}M"
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = subprocess.run(
            ["yt-dlp", "--get-id", f"ytsearch5:{query}"],
            capture_output=True, text=True, check=True
        )
        video_ids = result.stdout.strip().splitlines()[:2]  # Limit to 2 videos
        video_urls = [f"https://www.youtube.com/watch?v={vid_id}" for vid_id in video_ids]

        print("📄 Found video URLs:")
        for url in video_urls:
            print(url)

        for idx, url in enumerate(video_urls, 1):
            output_path = os.path.join(output_dir, f"source{idx}.mp4")
            print(f"\n⏳ Downloading #{idx}: {url}")
            try:
                subprocess.run([
                    "yt-dlp", "-f", "bestvideo[ext=mp4]",
                    "--max-filesize", MAX_FILESIZE,
                    "-o", output_path,
                    url
                ], check=True)
            except subprocess.CalledProcessError:
                print(f"⚠️  {url} too large or .mp4 not available — downloading first 10 minutes instead")
                try:
                    subprocess.run([
                        "yt-dlp", "-f", "bestvideo[ext=mp4]",
                        "--download-sections", "*00:00:00-00:10:00",
                        "-o", output_path,
                        url
                    ], check=False)
                except KeyboardInterrupt:
                    print("❌ Interrupted fallback download. Exiting.")
                    break
            except KeyboardInterrupt:
                print("❌ Skipped by user (Ctrl+C). Moving to next video.")
                continue

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to retrieve video URLs for query: {query}\n{e}")
    except KeyboardInterrupt:
        print("⛔ Interrupted during video retrieval. Aborting.")
    return len(video_urls)


# Function to extract frames from video
def extract_frames_from_videos(video_dir, frame_output_dir):
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 6:
            print(f"❗ {video_file} has too few frames, skipping.")
            continue

        # Pick 6 frames evenly spaced
        interval = total_frames // (6 + 1)
        frames_to_save = [interval * (i + 1) for i in range(6)]
        saved_count = 0
        
        for idx, frame_index in enumerate(frames_to_save):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame_filename = f"{video_file}_frame_{idx + 1}.jpg"
                frame_save_path = os.path.join(frame_output_dir, frame_filename)
                cv2.imwrite(frame_save_path, frame)
                saved_count += 1
        cap.release()
        print(f"✅ Extracted {saved_count} frames from {video_file}")
        
    #return len(video_files)

def process_and_store_images(image_dir):
    conn = get_db_connection()
    cursor = conn.cursor()

    for root, _, files in os.walk(image_dir):
        for image_file in files:
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, image_file)

                try:
                    # Validate image
                    with Image.open(image_path) as img:
                        img.verify()  
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")
                        img = img.resize((256, 256))
                        img.save(image_path)

                    with open(image_path, 'rb') as file:
                        image_data = file.read()

                    cursor.execute(
                        """
                        INSERT INTO images_crawled (image_id, image_data)
                        VALUES (%s, %s)
                        ON CONFLICT (image_id) DO NOTHING
                        """,
                        (image_file, psycopg2.Binary(image_data))
                    )
                    logging.info(f"✅ Stored {image_file} in PostgreSQL")

                except UnidentifiedImageError:
                    logging.warning(f"⚠️ Skipped invalid/corrupt image: {image_file}")
                except Exception as e:
                    logging.error(f"❌ Failed to process {image_file}: {e}")

    conn.commit()
    cursor.close()
    conn.close()
    logging.info("✅ Finished processing all images.")


# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 2),
    'retries': 1,
}

dag = DAG(
    'image_video_processing',
    default_args=default_args,
    description='A DAG to process images and videos from Google and YouTube',
    schedule_interval=None,  
    catchup = False, 
)

# Task Definition 
init_task = PythonOperator(
    task_id='init_schema',
    python_callable=init_schema,
    dag=dag
)

download_images = PythonOperator(
    task_id='download_images',
    python_callable=download_images_from_google,
    op_args=["street webcam car", "/tmp/airflow_data/images"],  # Search query and output directory
    dag=dag
)

download_videos = PythonOperator(
    task_id='download_videos',
    python_callable=download_videos_from_youtube,
    op_args=["traffic camera video", "/tmp/airflow_data/videos"],  # Search query and output directory
    dag=dag
)

extract_frames = PythonOperator(
    task_id='extract_frames',
    python_callable=extract_frames_from_videos,
    op_args=["/tmp/airflow_data/videos", "/tmp/airflow_data/frames"],
    dag=dag
)

process_images = PythonOperator(
    task_id='process_images',
    python_callable=process_and_store_images,
    op_args=["/tmp/airflow_data"],
    dag=dag
)

init_task >> download_images >> download_videos >> extract_frames >> process_images

