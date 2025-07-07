FROM python:3.11-slim

WORKDIR /app

# Install audio system dependencies (all 100% required by librosa + soundfile + resemblyzer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libsndfile1-dev \
    libasound2 \
    libasound2-dev \
    ffmpeg \
    portaudio19-dev \
    libportaudio2 \
    libpulse-dev \
    libjack-jackd2-dev \
    gcc \
    g++ \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*


# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Avoid permission issues in some containers
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib




# Install Python deps
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy your actual code
COPY src/ .

# Expose port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
