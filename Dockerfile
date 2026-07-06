FROM python:3.12-slim

# DejaVu Sans Mono is the glyph renderer's preferred fallback font
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app

WORKDIR /app

# Build from the committed lockfile so the container resolves the exact
# same dependency graph as CI and dev environments.
COPY pyproject.toml uv.lock README.MD LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir "uv>=0.11,<0.13" \
    && uv sync --frozen --no-dev --all-extras \
    && pip uninstall -y -q uv

ENV PATH="/app/.venv/bin:$PATH"

# Volume-mount target for one-shot CLI runs:
#   docker run --rm -v "$(pwd):/data" ascii-magic image photo.png -o art.txt
RUN mkdir /data && chown app:app /data
WORKDIR /data
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).status == 200 else 1)"]

# Default command serves the web GUI; any other args run the unified CLI.
ENTRYPOINT ["ascii-magic"]
CMD ["web", "--host", "0.0.0.0", "--port", "8000"]
