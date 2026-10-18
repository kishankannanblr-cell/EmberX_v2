# Production GPU/CPU Container for Vertex AI Prediction
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    AIP_HTTP_PORT=8080

WORKDIR /app

# Install system dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY gcp/vertex_serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy prediction code and SAHI module
COPY sahi_obb_inference.py /app/sahi_obb_inference.py
COPY gcp/vertex_serving/main.py /app/main.py

# Optional: Copy default weights if baking into container image
COPY emberx_v2.pt /app/emberx_v2.pt

EXPOSE 8080

CMD ["python", "main.py"]
