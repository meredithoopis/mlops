FROM apache/airflow:2.10.5

# Switch to root to install OS packages
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libsm6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to airflow user before installing Python packages
USER airflow

RUN pip install --no-cache-dir \
      opencv-python-headless \
      psycopg2-binary \
      numpy \
      pandas
