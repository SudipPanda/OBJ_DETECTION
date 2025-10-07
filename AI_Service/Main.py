from fastapi import FastAPI , HTTPException , UploadFile , File
from ultralytics import YOLO
import numpy as np
import cv2
import base64


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
    for box in res.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        class_name = res.names[int(cls)]
        confidence = round(float(conf), 2)

        # Append detection info
        detections.append({
            "class_id": int(cls),
            "class_name": class_name,
            "confidence": confidence,
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

        # 🆕 Draw rectangle and label
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"{class_name} {confidence}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    output_path = "output.jpg"
    cv2.imwrite(output_path, img)

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")


    return {"detections": detections, "width": w, "height": h , "image_base64": img_base64}



