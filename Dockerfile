FROM python:3.12-slim

# DejaVu Sans Mono is the glyph renderer's preferred fallback font
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.MD LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"

EXPOSE 8000

CMD ["uvicorn", "ascii_magic.webapp:app", "--host", "0.0.0.0", "--port", "8000"]
