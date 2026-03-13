FROM python:3.11-slim

# Instalar ffmpeg e dependências
RUN apt-get update && apt-get install -y \
    ffmpeg curl wget \
    && rm -rf /var/lib/apt/lists/*

# Instalar yt-dlp NIGHTLY direto do GitHub (versão mais recente — contém fixes anti-bloqueio)
RUN wget -q https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -O /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 5000

CMD ["python3", "app.py"]
