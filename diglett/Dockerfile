FROM python:3.9-slim

WORKDIR /app

# Install uv
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

RUN uv pip install --system --no-cache fastapi uvicorn python-multipart websockets speechbrain numpy torch torchaudio

COPY ./main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
