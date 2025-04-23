import mlflow
from ultralytics import YOLO

mlflow.set_tracking_uri('http://localhost:5000')  # Thay đổi nếu dùng MLflow server khác

# Bắt đầu MLflow run
with mlflow.start_run(run_name="YOLO11-training"):

    # Ghi log các params chính
    mlflow.log_param("epochs", 20) 
    mlflow.log_param("imgsz", 416)
    mlflow.log_param("device", "cpu")
    mlflow.log_param("model", "yolo11n.pt")

    # Load pretrained YOLO model
    model = YOLO("yolo11n.pt")

    # Train
    results = model.train(
        data="yolo_config.yaml",
        epochs=20,  
        imgsz=416,
        device="cpu",
    )

    # Đánh giá model trên val
    metrics = model.val(data="yolo_config.yaml")

    # Kiểm tra lại cấu trúc của metrics và ghi log metrics chính xác
    if metrics and hasattr(metrics, 'box'):
        mlflow.log_metrics({
            "val/box_loss": metrics.box.loss,
            "val/cls_loss": metrics.cls.loss,
            "val/dfl_loss": metrics.dfl.loss,
            "val/mAP50": metrics.box.map50,
            "val/mAP50-95": metrics.box.map,
        })
    
    print("Training & Logging với MLflow hoàn tất.")
