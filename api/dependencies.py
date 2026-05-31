# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Chargement et mise a disposition du modele.

Le modele est charge UNE SEULE FOIS au demarrage de l'application
(via le pattern lifespan dans main.py) et conserve en memoire.
Aucun rechargement n'a lieu a chaque requete.

Reutilise src/claims_classifier/inference/loader.py — pas de duplication.
"""

import logging
from dataclasses import dataclass

import torch

from claims_classifier.data.dataset import LabelEncoder
from claims_classifier.data.vocab import Vocabulary
from claims_classifier.inference.loader import load_for_inference

logger = logging.getLogger(__name__)


# =============================================================================
# CONTENEUR DU MODELE EN MEMOIRE
# =============================================================================

@dataclass
class ModelBundle:
    """Regroupe tout ce qui est necessaire a l'inference, charge une fois."""

    model: torch.nn.Module
    vocab: Vocabulary
    label_encoder: LabelEncoder
    model_name: str
    weighted_f1: float
    device: torch.device


# Singleton module-level rempli au demarrage (lifespan) et lu par les endpoints.
_BUNDLE: ModelBundle | None = None


# =============================================================================
# CHARGEMENT / ACCES
# =============================================================================

def load_model() -> ModelBundle:
    """
    Charge le modele et ses artefacts en memoire.

    Appele une seule fois au demarrage de l'API (lifespan).
    Force l'inference sur CPU : adapte au deploiement (Docker, Spaces)
    et garantit un comportement reproductible.

    Returns:
        Le ModelBundle pret a l'emploi.

    Raises:
        FileNotFoundError : checkpoint ou artefacts absents.
        RuntimeError      : toute autre erreur de chargement.
    """
    global _BUNDLE

    device = torch.device("cpu")
    logger.info("Chargement du modele pour l'API (device=cpu)...")

    try:
        model, vocab, label_encoder, arch_name, best_val_f1 = load_for_inference(
            device=device
        )
    except FileNotFoundError:
        # Propage tel quel : message explicite deja fourni par le loader.
        logger.error("Checkpoint ou artefacts introuvables.")
        raise
    except Exception as exc:  # noqa: BLE001 — on enrichit puis on re-leve
        logger.exception("Echec du chargement du modele.")
        raise RuntimeError(f"Impossible de charger le modele : {exc}") from exc

    _BUNDLE = ModelBundle(
        model=model,
        vocab=vocab,
        label_encoder=label_encoder,
        model_name=arch_name,
        weighted_f1=best_val_f1,
        device=device,
    )

    logger.info(
        f"Modele charge : {arch_name.upper()} | "
        f"{label_encoder.num_classes} classes | "
        f"weighted_f1={best_val_f1:.4f}"
    )
    return _BUNDLE


def unload_model() -> None:
    """Libere le modele de la memoire (appele a l'arret de l'API)."""
    global _BUNDLE
    _BUNDLE = None
    logger.info("Modele decharge de la memoire.")


def get_model_bundle() -> ModelBundle:
    """
    Dependance FastAPI : retourne le ModelBundle charge.

    Returns:
        Le ModelBundle en memoire.

    Raises:
        RuntimeError : si le modele n'est pas encore charge.
    """
    if _BUNDLE is None:
        raise RuntimeError(
            "Le modele n'est pas charge. L'API n'a pas demarre correctement."
        )
    return _BUNDLE


def is_model_loaded() -> bool:
    """Indique si le modele est actuellement charge en memoire."""
    return _BUNDLE is not None
