from db import create_connection

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
        xmin FLOAT,
        xmax FLOAT,
        ymin FLOAT,
        ymax FLOAT,
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


# Check if the new data is already in the database:
# step 1: psql -h localhost -U postgres -d car_detection
# step 2: SELECT * FROM user_feedback ORDER BY timestamp DESC LIMIT 10;