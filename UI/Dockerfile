# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV, Gradio, etc.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all UI service files into container
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn gradio requests opencv-python-headless numpy

# Expose Gradio port
EXPOSE 7860

# Command to run FastAPI app (which launches Gradio on startup)
CMD ["uvicorn", "main_ui:app", "--host", "0.0.0.0", "--port", "7860"]
