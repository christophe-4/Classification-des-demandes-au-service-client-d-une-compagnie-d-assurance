# syntax=docker/dockerfile:1
#
# Image Docker de l'API Claims Classifier (CPU only).
#
# Build :
#   docker build -t claims-classifier-api .
# Run :
#   docker run -p 8000:8000 claims-classifier-api
#
# L'image embarque le checkpoint TextCNN et les artefacts (vocab,
# label_encoder) : elle est auto-suffisante, aucune dependance externe
# au runtime.

# =============================================================================
# STAGE 1 — Builder : installe les dependances dans un venv isole
# =============================================================================
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# venv dedie (copie tel quel dans le stage runtime)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# 1) torch CPU depuis l'index dedie (~190 Mo au lieu de ~2.5 Go avec CUDA)
RUN pip install --no-cache-dir \
        torch>=2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu

# 2) Le reste des dependances runtime depuis PyPI
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt


# =============================================================================
# STAGE 2 — Runtime : image finale minimale
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app" \
    CLAIMS_TRAINING__DEVICE=cpu

# Utilisateur non-root (securite)
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# venv prepare dans le builder
COPY --from=builder /opt/venv /opt/venv

# Code applicatif
COPY src/ ./src/
COPY api/ ./api/

# Artefacts du modele (checkpoint + vocab + label_encoder)
# PROJECT_ROOT = /app (calcule depuis src/claims_classifier/config.py),
# donc config.artifacts.models_dir -> /app/models et
# config.data.vocab_path -> /app/data/processed/vocab.json
COPY models/textcnn_best.pt ./models/textcnn_best.pt
COPY data/processed/vocab.json ./data/processed/vocab.json
COPY data/processed/label_encoder.json ./data/processed/label_encoder.json

# Droits a l'utilisateur non-root
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Sonde de sante : interroge /health (modele charge en memoire)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Lancement du serveur (pas de --reload en production)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
