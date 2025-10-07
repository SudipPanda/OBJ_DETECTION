from fastapi import FastAPI , HTTPException , UploadFile , File
from ultralytics import YOLO
import numpy as np
import cv2


app = FastAPI()
model = YOLO("models/yolov8n.pt")

@app.post("/detect")

async def detect(file: UploadFile=File()):
    #contents = await file.read()
    #return  len(UploadFile)
    try:
        content = await file.read()
        #Image_val = Image.open(io.BytesIO(content))
        #numpy_array  = np.array(Image_val)
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #result = model(numpy_array)[0]

    except Exception:
        raise HTTPException(status_code=500, detail="something went wrong")

    h, w = img.shape[:2]
    res = model(img)[0]
    detections = []
    for box in res.boxes.data.tolist():  # get all boxes as Python list
        x1, y1, x2, y2, conf, cls = box
        detections.append({
            "class_id": int(cls),
            "class_name": res.names[int(cls)],
            "confidence": round(float(conf), 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

    return {"detections": detections, "width": w, "height": h}








