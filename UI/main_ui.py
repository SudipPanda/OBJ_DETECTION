import os
import requests
from fastapi import FastAPI
import gradio as gr
import cv2
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://localhost:8000")

app = FastAPI(title="UI Backend - Gradio Interface")


# Function to send image to AI backend
def detect_objects(image):
    try:
        _, encoded_image = cv2.imencode(".jpg", image)
        files = {"file": ("image.jpg", encoded_image.tobytes(), "image/jpeg")}
        resp = requests.post(f"{AI_BACKEND_URL}/detect", files=files, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": resp.text}

    except Exception as e:
          return {"error": str(e)}


# Launch Gradio interface
def create_ui():


    iface = gr.Interface(
        fn=detect_objects,
        inputs=gr.Image(type="numpy"),
        outputs="json",
        title="Object Detection UI",
        description="Upload an image and get detected objects in JSON format."
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
