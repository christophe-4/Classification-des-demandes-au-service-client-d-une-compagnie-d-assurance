# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Application FastAPI — Claims Classifier API.

Lancement :
    uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Le modele est charge une seule fois au demarrage (pattern lifespan)
et conserve en memoire pour toutes les requetes.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.dependencies import load_model, unload_model
from api.routers import predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie de l'application.

    Au demarrage : charge le modele en memoire (une seule fois).
    A l'arret    : libere le modele.
    """
    logger.info("Demarrage de l'API — chargement du modele...")
    load_model()
    logger.info("API prete a recevoir des requetes.")
    yield
    logger.info("Arret de l'API — liberation des ressources...")
    unload_model()


app = FastAPI(
    title="Claims Classifier API",
    description="API de classification de réclamations clients (CFPB) — TextCNN from scratch",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware CORS — autorise les appels depuis une application web.
# En production, restreindre allow_origins a vos domaines de confiance.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints de prediction et de sante
app.include_router(predict.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirige la racine vers la documentation interactive (/docs)."""
    return RedirectResponse(url="/docs")


def run() -> None:
    """
    Point d'entree console (script `claims-api`).

    Lance le serveur uvicorn sur le port 8000. Pour le rechargement
    automatique en developpement, preferer :
        uv run uvicorn api.main:app --reload
    """
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
