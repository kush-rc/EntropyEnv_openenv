FROM python:3.10-slim

# ── OpenEnv labels (required for HF Space tagging) ──
LABEL org.opencontainers.image.title="multi-agent-dev-tools-env"
LABEL org.opencontainers.image.description="Multi-Agent Dev Tools RL Environment"
LABEL openenv="true"

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null || pip install --no-cache-dir \
    fastapi uvicorn pydantic openai requests packaging gradio python-dotenv

# Make sure results directory exists and is writable by any user
RUN mkdir -p results && chmod 777 results

# Copy project files
COPY . .

# Expose port 7860 (HuggingFace Spaces standard)
EXPOSE 7860

# Health check for HF Spaces
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
