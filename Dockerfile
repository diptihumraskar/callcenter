FROM python:3.11-slim

# ffmpeg is required by faster-whisper / mutagen for audio decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY app.py ./
RUN pip install --no-cache-dir .

COPY . .

RUN mkdir -p data/audio data/samples

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app.py"]
