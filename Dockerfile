FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg curl wget unzip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Instalar Deno
RUN curl -fsSL https://deno.land/install.sh | sh && \
    mv /root/.deno/bin/deno /usr/local/bin/deno && \
    chmod a+rx /usr/local/bin/deno

# Instalar yt-dlp mais recente
RUN wget -q "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp" \
    -O /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp \
    && yt-dlp --version

WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import cv2; print('OpenCV OK:', cv2.__version__)"

COPY app/ .

EXPOSE 5000
CMD ["python3", "app.py"]
