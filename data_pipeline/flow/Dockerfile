FROM apache/airflow:2.10.5

# Switch to root to install OS-level packages
USER root

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    psycopg2-binary \
    numpy \
    pandas \
    requests \
    python-dotenv \
    serpapi \
    tqdm \
    yt-dlp \
    google-search-results \
    Pillow \
    mlflow \
    ultralytics
