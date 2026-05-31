"""

Usage :
    uv run python scripts/train.py --model mlp
    uv run python scripts/train.py --model textcnn
    uv run python scripts/train.py --model mlp --epochs 10 --batch-size 128

Pipeline complet :
  1. Chargement et nettoyage des donnees
  2. Splits stratifies train/val/test
  3. Construction du vocabulaire et du label encoder
  4. Creation des DataLoaders
  5. Initialisation du modele
  6. Entrainement avec early stopping
  7. Evaluation finale sur le jeu de test
  8. Generation des figures et du rapport
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import typer

# Ajouter src/ au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claims_classifier.config import config
from claims_classifier.data.cleaning import run_cleaning
from claims_classifier.data.dataset import (
    LabelEncoder,
    make_loaders,
    make_splits,
)
from claims_classifier.data.loader import load_raw
from claims_classifier.data.vocab import Vocabulary
from claims_classifier.evaluation.metrics import check_objective, evaluate
from claims_classifier.evaluation.reports import (
    plot_confusion_matrix,
    plot_training_history,
    save_text_report,
)
from claims_classifier.models.mlp import MLP
from claims_classifier.models.textcnn import TextCNN
from claims_classifier.training.losses import build_loss
from claims_classifier.training.trainer import Trainer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Entrainement du classifieur de reclamations CFPB.")


@app.command()
def train(
    model: str = typer.Option(
        "textcnn",
        "--model",
        "-m",
        help="Architecture : 'mlp' ou 'textcnn'.",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Nombre max d'epoques (defaut : config).",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Taille du batch (defaut : config).",
    ),
) -> None:
    """Lance l'entrainement complet et evalue sur le jeu de test."""

    # Override config si arguments CLI fournis
    if epochs is not None:
        config.training.num_epochs = epochs
    if batch_size is not None:
        config.training.batch_size = batch_size

    if model not in ("mlp", "textcnn"):
        typer.echo(f"Modele inconnu : '{model}'. Choisir 'mlp' ou 'textcnn'.")
        raise typer.Exit(1)

    logger.info(f"Modele selectionne : {model.upper()}")

    # ── 0. Reproductibilite ───────────────────────────────────────────────────
    from claims_classifier.utils import set_seed

    set_seed(config.training.seed)

    # ── 1. Chargement et nettoyage ────────────────────────────────────────────
    logger.info("Etape 1/8 : Chargement et nettoyage des donnees")
    df = load_raw()
    df = run_cleaning(df)

    # ── 2. Splits stratifies ──────────────────────────────────────────────────
    logger.info("Etape 2/8 : Splits stratifies train/val/test")
    train_df, val_df, test_df = make_splits(df)

    # ── 3. Vocabulaire et label encoder ──────────────────────────────────────
    logger.info("Etape 3/8 : Construction vocabulaire et label encoder")
    vocab = Vocabulary.build(train_df["text"])
    vocab.save()

    label_encoder = LabelEncoder.build(train_df["label"])
    label_encoder.save()

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    logger.info("Etape 4/8 : Creation des DataLoaders")
    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, vocab, label_encoder
    )

    # ── 5. Modele ─────────────────────────────────────────────────────────────
    logger.info("Etape 5/8 : Initialisation du modele")
    vocab_size = vocab.size
    num_classes = label_encoder.num_classes

    if model == "mlp":
        net = MLP(vocab_size=vocab_size, num_classes=num_classes)
    else:
        net = TextCNN(vocab_size=vocab_size, num_classes=num_classes)

    logger.info(
        f"Parametres entrainables : {sum(p.numel() for p in net.parameters() if p.requires_grad):,}"
    )

    # ── 6. Loss ponderee ──────────────────────────────────────────────────────
    logger.info("Etape 6/8 : Construction de la loss ponderee")
    device = torch.device(
        "cuda" if config.training.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    loss_fn = build_loss(train_df["label"], label_encoder, device)

    # ── 7. Entrainement ───────────────────────────────────────────────────────
    logger.info("Etape 7/8 : Entrainement")
    trainer = Trainer(
        model=net,
        loss_fn=loss_fn,
        num_classes=num_classes,
        model_name=model,
        device=device,
    )
    history = trainer.fit(train_loader, val_loader)

    # Courbes d'entrainement
    plot_training_history(history, model_name=model)

    # ── 8. Evaluation finale sur le test ─────────────────────────────────────
    logger.info("Etape 8/8 : Evaluation finale sur le jeu de test")
    results = evaluate(net, test_loader, label_encoder, device)
    results.print_summary(f"Resultats FINAUX — {model.upper()} (jeu de TEST)")

    # Matrice de confusion (livrable brief)
    plot_confusion_matrix(results, label_encoder.classes, model_name=model)

    # Rapport texte
    extra = (
        f"Modele          : {model.upper()}\n"
        f"Parametres      : {sum(p.numel() for p in net.parameters()):,}\n"
        f"Epochs effectuees : {len(history['train_loss'])}\n"
        f"Device          : {device}\n"
    )
    save_text_report(results, model_name=model, extra_info=extra)

    # Verification objectif brief
    passed = check_objective(results)

    if passed:
        typer.echo(f"\n Weighted F1 = {results.weighted_f1:.4f} — Objectif 75% ATTEINT")
    else:
        typer.echo(
            f"\n Weighted F1 = {results.weighted_f1:.4f} — "
            f"Objectif 75% NON ATTEINT — "
            f"Ajuster les hyperparametres"
        )


if __name__ == "__main__":
    app()
