FROM python:3.11-slim

# Instalar dependências base
RUN apt-get update && apt-get install -y \
    ffmpeg curl wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Instalar Deno (runtime JS necessário para o yt-dlp resolver o n-challenge do YouTube)
RUN curl -fsSL https://deno.land/install.sh | sh && \
    mv /root/.deno/bin/deno /usr/local/bin/deno && \
    chmod a+rx /usr/local/bin/deno

# Instalar yt-dlp mais recente
RUN wget -q https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -O /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

# Configurar yt-dlp para usar Deno
RUN mkdir -p /etc/yt-dlp && \
    echo '--js-runtimes deno' > /etc/yt-dlp/yt-dlp.conf

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 5000

CMD ["python3", "app.py"]
