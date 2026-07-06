FROM python:3.12-slim

# DejaVu Sans Mono is the glyph renderer's preferred fallback font
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.MD LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"

# Volume-mount target for one-shot CLI runs:
#   docker run --rm -v "$(pwd):/data" ascii-magic image photo.png -o art.txt
WORKDIR /data

EXPOSE 8000

# Default command serves the web GUI; any other args run the unified CLI.
ENTRYPOINT ["ascii-magic"]
CMD ["web", "--host", "0.0.0.0", "--port", "8000"]
