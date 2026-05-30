"""
Fonctions utilitaires transverses.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Fixe toutes les graines aleatoires pour la reproductibilite.

    Couvre :
      - random  (Python standard)
      - numpy   (operations vectorielles)
      - torch   (init des poids, dropout)
      - torch.cuda (si GPU disponible)

    Garantit qu'un meme run avec la meme seed produit des resultats identiques.
    Indispensable pour un projet reproductible (cours : reproductibilite).

    Args:
        seed: Valeur de la graine (ex: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Comportement deterministe sur GPU (legerement plus lent)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Seed fixee a {seed} (random, numpy, torch, cuda)")


def get_device() -> torch.device:
    """
    Selectionne le dispositif de calcul (GPU si dispo, sinon CPU).

    Returns:
        torch.device approprie.
    """
    from claims_classifier.config import config

    if config.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Dispositif : GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("Dispositif : CPU")
    return device