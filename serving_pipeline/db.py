import psycopg2
from dotenv import load_dotenv
import os 

dotenv_path = os.path.abspath(os.path.join("..", "data_pipeline", "flow", ".env"))
load_dotenv(dotenv_path=dotenv_path)
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT", 5432)) 


def create_connection():
    conn = psycopg2.connect(
        host=DB_HOST, 
        database=DB_NAME, 
        user=DB_USER, 
        password=DB_PASSWORD, 
        port=DB_PORT
    )
    return conn

def create_table():
    conn = create_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_feedback (
        id SERIAL PRIMARY KEY,
        image_id TEXT,
        source TEXT,
        label_name TEXT,
        confidence FLOAT,
        x_min FLOAT,
        x_max FLOAT,
        y_min FLOAT,
        y_max FLOAT,
        is_occluded BOOLEAN,
        is_truncated BOOLEAN,
        is_group_of BOOLEAN,
        is_depiction BOOLEAN,
        is_inside BOOLEAN,
        xclick1x FLOAT,
        xclick2x FLOAT,
        xclick3x FLOAT,
        xclick4x FLOAT,
        xclick1y FLOAT,
        xclick2y FLOAT,
        xclick3y FLOAT,
        xclick4y FLOAT,
        labelname_text TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    cur.close()
    conn.close()

def save_full_feedback(
    image_id, source, label_name, confidence,
    x_min, x_max, y_min, y_max,
    is_occluded, is_truncated, is_group_of, is_depiction, is_inside,
    xclick1x, xclick2x, xclick3x, xclick4x,
    xclick1y, xclick2y, xclick3y, xclick4y,
    labelname_text
):
    
    is_occluded = bool(is_occluded)
    is_truncated = bool(is_truncated)
    is_group_of = bool(is_group_of)
    is_depiction = bool(is_depiction)
    is_inside = bool(is_inside)

    # Connect wth the database 
    conn = create_connection()
    cur = conn.cursor()

    # Adding data to the database
    cur.execute("""
        INSERT INTO user_feedback (
            image_id, source, label_name, confidence,
            x_min, x_max, y_min, y_max,
            is_occluded, is_truncated, is_group_of, is_depiction, is_inside,
            xclick1x, xclick2x, xclick3x, xclick4x,
            xclick1y, xclick2y, xclick3y, xclick4y,
            labelname_text
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        image_id, source, label_name, confidence,
        x_min, x_max, y_min, y_max,
        is_occluded, is_truncated, is_group_of, is_depiction, is_inside,
        xclick1x, xclick2x, xclick3x, xclick4x,
        xclick1y, xclick2y, xclick3y, xclick4y,
        labelname_text
    ))

    conn.commit()
    cur.close()
    conn.close()
