FROM apache/airflow:2.7.3

# Installing packages as root 
USER root
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
RUN python -m pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --timeout=200 --no-cache-dir --upgrade -r requirements.txt
COPY . .

EXPOSE 8501

ENTRYPOINT []
CMD ["streamlit", "run", "serving_pipeline/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]