FROM python:3.10-slim
WORKDIR /app

# System dependencies needed for OpenCV & Ultralytics
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn ultralytics opencv-python-headless numpy python-multipart

EXPOSE 8000

CMD ["uvicorn", "Main:app", "--host", "0.0.0.0", "--port", "8000"]
