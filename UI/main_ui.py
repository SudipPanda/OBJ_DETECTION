import os
import requests
from fastapi import FastAPI
import gradio as gr
import cv2
import numpy as np
import base64
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://localhost:8000")

app = FastAPI(title="UI Backend - Gradio Interface")


# Function to send image to AI backend
def detect_objects(image):
    try:
        _, encoded_image = cv2.imencode(".jpg", image)
        files = {"file": ("image.jpg", encoded_image.tobytes(), "image/jpeg")}
        resp = requests.post(f"{AI_BACKEND_URL}/detect", files=files, timeout=30)

        if resp.status_code != 200:
            return None, {"error": resp.text}

        data = resp.json()

        # Decode base64 image if provided
        if "image_base64" in data:
            img_data = base64.b64decode(data["image_base64"])
            np_img = np.frombuffer(img_data, np.uint8)
            annotated_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        else:
            annotated_img = image  # fallback

        return annotated_img, data["detections"]

    except Exception as e:
        return None, {"error": str(e)}


# Launch Gradio interface
def create_ui():
    iface = gr.Interface(
        fn=detect_objects,
        inputs=gr.Image(type="numpy", label="Upload Image"),
        outputs=[
            gr.Image(type="numpy", label="Detected Image"),
            gr.JSON(label="Detection Results")
        ],
        title="🧠 Object Detection UI",
        description="Upload an image to detect objects and view results with bounding boxes."
    )

    return iface


@app.on_event("startup")
def startup_event():
    # Launch Gradio inside FastAPI
    ui = create_ui()
    ui.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, inline=False, inbrowser=False)


@app.get("/health")
def health():
      return {"status": "ok", "ai_backend": AI_BACKEND_URL}
