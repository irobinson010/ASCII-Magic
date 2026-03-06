FROM python:3.13-slim

# Install the DejaVu monospace font (required for glyph rendering)
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for user data (volume mount target)
WORKDIR /data

# Copy project files and install
COPY pyproject.toml README.MD LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir .

ENTRYPOINT ["ascii-magic"]
