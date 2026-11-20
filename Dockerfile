# --- Stage 1: train the model inside the build so the image is self-contained ---
FROM python:3.12-slim AS train
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY data ./data
RUN survival-predictor train

# --- Stage 2: slim runtime image serving the API ---
FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home appuser
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && rm -rf src
COPY --from=train /app/models ./models
USER appuser
ENV MODEL_PATH=/app/models/model.joblib
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "survival_predictor.api:app", "--host", "0.0.0.0", "--port", "8000"]
