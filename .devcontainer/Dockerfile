FROM mcr.microsoft.com/devcontainers/python:3.11

# Install minimal system dependencies useful for Python builds and testing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is current
RUN python -m pip install --upgrade pip

USER vscode
