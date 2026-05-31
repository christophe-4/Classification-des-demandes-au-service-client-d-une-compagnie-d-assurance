# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Logging RGPD-conforme des predictions en production.

RGPD : aucun texte brut n'est jamais enregistre.
On conserve uniquement des metadonnees derivees :
  - longueur du texte (en mots)
  - nombre de tokens inconnus
  - classe predite et probabilite
  - informations de performance

Thread-safe : utilise un verrou global pour les appends concurrents.
Format : JSONL (une prediction JSON par ligne) — portable, sans DB externe.
"""

import json
import logging
import threading
from datetime import datetime, timezone

from claims_classifier.config import config

logger = logging.getLogger(__name__)

# Verrou global — garantit les appends atomiques en contexte multithread
_LOCK = threading.Lock()


def log_prediction(
    text_length: int,
    num_unknown_tokens: int,
    predicted_class: str,
    confidence: float,
    top_k: list[dict],
    inference_time_ms: float,
    model_name: str,
) -> None:
    """
    Enregistre les metadonnees d'une prediction dans le fichier JSONL.

    RGPD : le texte brut n'est JAMAIS enregistre — uniquement des metadonnees
    derivees qui ne permettent pas de retrouver la reclamation originale.

    Args:
        text_length        : Nombre de mots dans le texte nettoye.
        num_unknown_tokens : Nombre de mots hors vocabulaire (signal de derive).
        predicted_class    : Classe predite (ex: "credit_reporting").
        confidence         : Probabilite de la classe principale (0.0 - 1.0).
        top_k              : Liste de dicts {"class_name": str, "probability": float}.
        inference_time_ms  : Temps d'inference en millisecondes.
        model_name         : Architecture utilisee (ex: "textcnn").

    Raises:
        Ne leve aucune exception — les erreurs sont loggees mais ne bloquent pas.
    """
    cfg = config.monitoring
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    unk_rate = round(num_unknown_tokens / max(text_length, 1), 4)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_length": text_length,
        "num_unknown_tokens": num_unknown_tokens,
        "unk_rate": unk_rate,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "top_k": [
            {"class_name": item["class_name"], "probability": round(item["probability"], 4)}
            for item in top_k
        ],
        "inference_time_ms": round(inference_time_ms, 2),
        "model_name": model_name,
    }

    line = json.dumps(record, ensure_ascii=False) + "\n"

    with _LOCK:
        with open(cfg.predictions_log_path, "a", encoding="utf-8") as fh:
            fh.write(line)


def load_logs(n: int | None = None) -> list[dict]:
    """
    Charge les logs de predictions depuis le fichier JSONL.

    Args:
        n : Si fourni, retourne uniquement les n derniers enregistrements.

    Returns:
        Liste de dicts (predictions, du plus ancien au plus recent).
        Liste vide si le fichier n'existe pas encore.
    """
    log_path = config.monitoring.predictions_log_path
    if not log_path.exists():
        return []

    records = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Ligne JSONL invalide ignoree : {line[:80]}")

    if n is not None:
        records = records[-n:]

    return records
