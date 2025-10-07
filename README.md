AI Object Detection Microservice
A fully Dockerized AI Object Detection System built using FastAPI, Gradio, and Ultralytics YOLO.
AI Service (FastAPI) — Handles image processing and object detection.
UI Service (Gradio + FastAPI) — Provides a clean web interface for uploading images and viewing detection results.

project Structure :-

AI_Object_Detection
│
├── AI_Service
│   ├── Main.py             
│     
│   └── Dockerfile            
│
├── UI_Service
│   ├── main.py               
│   ├     
│   └── Dockerfile            
│
├── docker-compose.yml        
└── README.md        

TO BUILD AND START ALL THE SERVICE:-
docker-compose up --build

ACCESS THE APPLICATIONS:-
UI Frontend (Gradio) → http://localhost:7860


