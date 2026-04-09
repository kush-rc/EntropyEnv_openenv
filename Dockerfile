# Stage 1: Build dependencies
FROM python:3.10-slim AS builder
WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir . || pip install --no-cache-dir fastapi uvicorn pydantic openai requests packaging gradio python-dotenv

# Stage 2: Runtime
FROM python:3.10-slim
LABEL org.opencontainers.image.title="multi-agent-dev-tools-env"
LABEL org.opencontainers.image.description="Multi-Agent Dev Tools RL Environment"
LABEL openenv="true"

WORKDIR /app

# Copy installed packages AND scripts (uvicorn binary) from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY . .

# Results directory
RUN mkdir -p results && chmod 777 results

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
