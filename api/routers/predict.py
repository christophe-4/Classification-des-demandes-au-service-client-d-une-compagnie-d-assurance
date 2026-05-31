# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Endpoints de l'API : /health et /predict.

Le pipeline d'inference reutilise exactement les briques du projet :
  - clean_text (data/cleaning.py) pour le nettoyage
  - vocab.encode (data/vocab.py) pour la tokenisation
  - le meme padding/troncature que ClaimsDataset
Aucune logique n'est dupliquee.
"""

import logging
import time

import torch
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import ModelBundle, get_model_bundle, is_model_loaded
from api.schemas import (
    HealthResponse,
    PredictionItem,
    PredictRequest,
    PredictResponse,
)
from claims_classifier.config import config
from claims_classifier.data.cleaning import clean_text
from claims_classifier.monitoring.logger import log_prediction

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# /health
# =============================================================================


@router.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health(bundle: ModelBundle = Depends(get_model_bundle)) -> HealthResponse:
    """
    Verifie l'etat de l'API et du modele.

    Retourne le statut de service, l'architecture chargee, le nombre de
    classes et le Weighted F1 du checkpoint. Utile pour les sondes de
    liveness/readiness (Docker, Kubernetes).
    """
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        model_name=bundle.model_name,
        num_classes=bundle.label_encoder.num_classes,
        weighted_f1=round(bundle.weighted_f1, 4),
    )


# =============================================================================
# /predict
# =============================================================================


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(
    request: PredictRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
) -> PredictResponse:
    """
    Classifie une reclamation client en categories financieres.

    Applique le pipeline complet : nettoyage du texte, tokenisation,
    inference TextCNN, softmax, puis renvoie les `top_k` categories les
    plus probables avec leurs probabilites.

    Erreurs :
      - 422 si le texte devient vide apres nettoyage (ponctuation seule, etc.).
    """
    # Nettoyage identique au pipeline d'entrainement
    cleaned = clean_text(request.text)
    if not cleaned or not cleaned.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Texte vide apres nettoyage : aucun mot exploitable.",
        )

    vocab = bundle.vocab
    label_encoder = bundle.label_encoder
    max_len = config.preprocessing.max_seq_length

    # Metadonnees RGPD — calculees ici, jamais le texte brut
    words = cleaned.split()
    text_length = len(words)
    num_unknown_tokens = sum(1 for w in words if w not in vocab.word2idx)

    # Tokenisation + troncature + padding (identique a ClaimsDataset)
    ids = vocab.encode(cleaned)[:max_len]
    ids = ids + [vocab.pad_id] * (max_len - len(ids))
    input_ids = torch.tensor([ids], dtype=torch.long, device=bundle.device)

    # Inference chronometree
    start = time.perf_counter()
    with torch.no_grad():
        logits = bundle.model(input_ids)
        probs = torch.softmax(logits, dim=1)[0]
    inference_time_ms = (time.perf_counter() - start) * 1000.0

    # Top-k
    k = min(request.top_k, label_encoder.num_classes)
    top = torch.topk(probs, k)
    top_items = [
        PredictionItem(
            class_name=label_encoder.decode(int(idx)),
            probability=round(float(prob), 4),
        )
        for prob, idx in zip(top.values.tolist(), top.indices.tolist())
    ]

    response = PredictResponse(
        prediction=top_items[0].class_name,
        confidence=top_items[0].probability,
        top_k=top_items,
        model_name=bundle.model_name,
        weighted_f1=round(bundle.weighted_f1, 4),
        inference_time_ms=round(inference_time_ms, 2),
    )

    # Logging RGPD-conforme — fail-safe : une erreur ici ne bloque jamais la reponse
    try:
        log_prediction(
            text_length=text_length,
            num_unknown_tokens=num_unknown_tokens,
            predicted_class=response.prediction,
            confidence=response.confidence,
            top_k=[{"class_name": item.class_name, "probability": item.probability} for item in top_items],
            inference_time_ms=response.inference_time_ms,
            model_name=bundle.model_name,
        )
    except Exception as log_err:
        logger.warning(f"Logging monitoring echoue (non-bloquant) : {log_err}")

    return response
