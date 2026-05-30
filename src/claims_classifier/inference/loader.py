"""
Chargement autonome d'un checkpoint pour l'inference (Phase 2).

Utilise par evaluate.py et predict.py sans passer par le Trainer.
Avantage : pas besoin d'une loss_fn ni d'un SummaryWriter TensorBoard —
           on charge uniquement les poids et les artefacts necessaires.

Usage :
    from claims_classifier.inference.loader import load_for_inference

    model, vocab, label_encoder, arch_name, best_val_f1 = load_for_inference()
    model, vocab, label_encoder, arch_name, best_val_f1 = load_for_inference("textcnn")
"""

import logging
from pathlib import Path
from typing import Optional

import torch

from claims_classifier.config import config
from claims_classifier.data.dataset import LabelEncoder
from claims_classifier.data.vocab import Vocabulary
from claims_classifier.models.mlp import MLP
from claims_classifier.models.textcnn import TextCNN
from claims_classifier.utils import get_device

logger = logging.getLogger(__name__)


def load_for_inference(
    model_name: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Charge un checkpoint et retourne le modele pret pour l'inference.

    Logique de selection du checkpoint :
      1. Si checkpoint_path est fourni, il est utilise directement.
      2. Sinon, si model_name est fourni, on cherche models/<model_name>_best.pt.
      3. Sinon, preference textcnn > mlp (meilleur modele par defaut).

    Args:
        model_name      : 'mlp' ou 'textcnn' (optionnel).
        checkpoint_path : Chemin explicite vers un checkpoint .pt (optionnel).
        device          : Dispositif de calcul (optionnel, auto-detecte sinon).

    Returns:
        Tuple (model, vocab, label_encoder, arch_name, best_val_f1) ou :
          - model         : nn.Module en mode eval, sur device.
          - vocab         : Vocabulary charge depuis data/processed/vocab.json.
          - label_encoder : LabelEncoder charge depuis data/processed/label_encoder.json.
          - arch_name     : 'mlp' ou 'textcnn' (lu depuis le checkpoint).
          - best_val_f1   : Meilleur Weighted F1 de validation (float).

    Raises:
        FileNotFoundError : Aucun checkpoint trouve.
        ValueError        : Architecture inconnue dans le checkpoint.
    """
    device = device if device is not None else get_device()

    # ── Resolution du chemin ─────────────────────────────────────────────────
    if checkpoint_path is None:
        models_dir = config.artifacts.models_dir
        if model_name is not None:
            checkpoint_path = models_dir / f"{model_name}_best.pt"
        else:
            textcnn_path = models_dir / "textcnn_best.pt"
            mlp_path = models_dir / "mlp_best.pt"
            if textcnn_path.exists():
                checkpoint_path = textcnn_path
            elif mlp_path.exists():
                checkpoint_path = mlp_path
            else:
                raise FileNotFoundError(
                    f"Aucun checkpoint trouve dans {models_dir}. "
                    "Lancez d'abord l'entrainement : "
                    "uv run python scripts/train.py"
                )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}\n"
            "Lancez d'abord l'entrainement : uv run python scripts/train.py"
        )

    # ── Chargement du checkpoint ─────────────────────────────────────────────
    logger.info(f"Chargement du checkpoint : {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    arch_name: str = checkpoint["model_name"]
    num_classes: int = checkpoint["num_classes"]
    best_val_f1: float = checkpoint["best_val_f1"]

    logger.info(
        f"Checkpoint charge : {arch_name.upper()} | "
        f"num_classes={num_classes} | "
        f"best_val_f1={best_val_f1:.4f}"
    )

    # ── Artefacts (vocab + label_encoder) ────────────────────────────────────
    vocab = Vocabulary.load()
    label_encoder = LabelEncoder.load()
    vocab_size = len(vocab)

    # ── Reconstruction du modele ─────────────────────────────────────────────
    if arch_name == "mlp":
        model = MLP(vocab_size=vocab_size, num_classes=num_classes)
    elif arch_name == "textcnn":
        model = TextCNN(vocab_size=vocab_size, num_classes=num_classes)
    else:
        raise ValueError(
            f"Architecture inconnue dans le checkpoint : '{arch_name}'"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Modele {arch_name.upper()} pret pour l'inference")

    return model, vocab, label_encoder, arch_name, best_val_f1
