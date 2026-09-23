# DOOM-FlyWire Dockerfile for Hugging Face Spaces & Containerized Deployments
FROM python:3.10-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential build tools, ViZDoom dependencies, and OpenGL/sound libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libsdl2-dev \
    libopenal-dev \
    zlib1g-dev \
    libbz2-dev \
    libjpeg-dev \
    libfluidsynth-dev \
    libgme-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (UID 1000) for Hugging Face Spaces security standards
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:/usr/local/bin:/usr/bin:/bin

WORKDIR /home/user/app

# Install Python dependencies as user
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy project files with proper ownership
COPY --chown=user:user . .

# Hugging Face Spaces standard port is 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "server.py"]
