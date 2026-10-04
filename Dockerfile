FROM python:3.11-slim

# ffmpeg is required by faster-whisper / mutagen for audio decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# constraints.txt pins every dependency to the tested versions; without it pip
# backtracks through hundreds of google-genai / fastapi releases (10+ min builds).
COPY pyproject.toml constraints.txt README.md ./
COPY src ./src
COPY app.py ./
RUN pip install --no-cache-dir -c constraints.txt ".[observability]"

COPY . .

RUN mkdir -p data/audio data/samples

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app.py"]
