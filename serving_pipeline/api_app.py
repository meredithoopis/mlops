from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from PIL import Image
import io
from serving_pipeline import get_boxes_for_image

app = FastAPI(title="CSV Bounding Box API")

@app.get("/")
def read_root():
    return {"message": "CSV Bounding Box API is running"}

@app.post("/detect/")
async def detect_from_image(file: UploadFile = File(...)):
    filename = file.filename
    image_id = os.path.splitext(filename)[0]

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    boxes = get_boxes_for_image(image_id, w, h)

    result = []
    for (x1, y1), (x2, y2), label in boxes:
        result.append({
            "label": label,
            "box": [x1, y1, x2, y2]
        })

    return JSONResponse(content={"image_id": image_id, "detections": result})

#uvicorn serving_pipeline.api_app:app --reload
