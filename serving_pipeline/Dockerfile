FROM apache/airflow:2.7.3

# Install system dependencies
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
USER airflow
RUN python -m pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy app code
WORKDIR /app
COPY requirements.txt .
RUN pip install --timeout=200 --no-cache-dir --upgrade -r requirements.txt
COPY . .

# Run Streamlit app
EXPOSE 8501
CMD ["streamlit", "run", "serving_pipeline/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
