from ultralytics import YOLO

model = YOLO('/mlartifacts/0/16fa8bd85843416f930b613acbdd3b58/artifacts/weights/best.pt')
model.export(format='onnx')
