# File: Dockerfile

FROM ultralytics/ultralytics:8.3.122-cpu

RUN pip install pandas mlflow

WORKDIR /app
COPY . .

CMD ["python", "tracking_pipeline.py"]
