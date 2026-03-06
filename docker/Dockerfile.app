# docker/Dockerfile.app
# ======================
# Combined image: pre-built React frontend + Flask API backend.
# A single Flask process serves both the REST API and the static SPA.
#
# Build:
#   docker build -t vlsi-ai-app -f docker/Dockerfile.app .
#
# Run:
#   docker run --rm -p 5000:5000 vlsi-ai-app
#   # Then open http://localhost:5000 in a browser.
#
# Customise the port:
#   docker run --rm -p 8080:8080 -e PORT=8080 -e HOST=0.0.0.0 vlsi-ai-app
#
# Persist results between runs:
#   docker run --rm -p 5000:5000 \
#     -v $PWD/results:/app/results \
#     -v $PWD/datasets:/app/datasets \
#     vlsi-ai-app

# ---------------------------------------------------------------------------
# Stage 1 — build the React/Vite frontend
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend

# Install dependencies (package-lock.json ensures reproducible installs)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source and build
COPY frontend/ ./
RUN npm run build
# Output is in /build/frontend/dist

# ---------------------------------------------------------------------------
# Stage 2 — Python runtime with ngspice + the built frontend
# ---------------------------------------------------------------------------
FROM python:3.11-slim

LABEL org.opencontainers.image.title="vlsi-ai-app" \
      org.opencontainers.image.description="VLSI-AI Design Studio — frontend + backend combined" \
      org.opencontainers.image.source="https://github.com/j143/vlsi-ai-experiments"

# System dependencies: ngspice + C build tools (needed by some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ngspice \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (separate layer for better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project source
COPY . /app

# Install the vlsi-ai package (provides the CLI entry-point)
RUN pip install --no-cache-dir -e .

# Copy the pre-built React app from the frontend-builder stage
COPY --from=frontend-builder /build/frontend/dist /app/frontend/dist

# Ensure output directories exist
RUN mkdir -p /app/datasets /app/results

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
ENV VLSI_AI_DATASETS=/app/datasets \
    VLSI_AI_RESULTS=/app/results \
    STATIC_DIR=/app/frontend/dist \
    HOST=0.0.0.0 \
    PORT=5000

EXPOSE 5000

# Start the combined Flask server (serves API + SPA)
CMD ["python", "api/server.py"]
