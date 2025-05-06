import psycopg2

def create_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="car_detection",
        user="postgres",
        password="admin", # use ur own password
        port=5432
    )
    return conn

def save_full_feedback(
    image_id, source, label_name, confidence,
    x_min, x_max, y_min, y_max,
    is_occluded, is_truncated, is_group_of, is_depiction, is_inside,
    xclick1x, xclick2x, xclick3x, xclick4x,
    xclick1y, xclick2y, xclick3y, xclick4y,
    labelname_text
):
    # Chuyển đổi các giá trị integer 0 và 1 thành True/False cho các cột boolean
    is_occluded = bool(is_occluded)
    is_truncated = bool(is_truncated)
    is_group_of = bool(is_group_of)
    is_depiction = bool(is_depiction)
    is_inside = bool(is_inside)

    # Kết nối với cơ sở dữ liệu
    conn = create_connection()
    cur = conn.cursor()

    # Chạy câu lệnh SQL với đúng kiểu dữ liệu
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

