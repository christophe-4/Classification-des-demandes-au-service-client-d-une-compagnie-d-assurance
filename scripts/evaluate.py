"""
Evaluation du meilleur modele entraine sur le jeu de test.

Usage :
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --model textcnn
    uv run python scripts/evaluate.py --model mlp

Produit :
  - Metriques : Weighted F1, Macro F1, Accuracy, F1 par classe
  - Matrice de confusion   : reports/figures/confusion_matrix_<model>.png
  - Rapport texte complet  : reports/report_<model>.txt
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claims_classifier.config import config
from claims_classifier.data.cleaning import run_cleaning
from claims_classifier.data.dataset import ClaimsDataset, make_splits
from claims_classifier.data.loader import load_raw
from claims_classifier.evaluation.metrics import check_objective, evaluate
from claims_classifier.evaluation.reports import (
    plot_confusion_matrix,
    save_text_report,
)
from claims_classifier.inference.loader import load_for_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Evaluation sur le jeu de test.")


@app.command()
def evaluate_model(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Architecture : 'mlp' ou 'textcnn' (defaut : meilleur disponible).",
    ),
) -> None:
    """Evalue le meilleur modele entraine sur le jeu de test officiel."""

    # ── Etape 1 : chargement du modele ───────────────────────────────────────
    logger.info("Etape 1/4 : Chargement du checkpoint")
    model_obj, vocab, label_encoder, arch_name, best_val_f1 = load_for_inference(
        model_name=model
    )
    device = next(model_obj.parameters()).device

    # ── Etape 2 : reconstruction du jeu de test ──────────────────────────────
    # On rejoue le meme pipeline qu'a l'entrainement avec la meme seed (42).
    # Le split etant deterministe, on obtient exactement le meme test set.
    logger.info("Etape 2/4 : Reconstruction du jeu de test (seed=%d)", config.split.seed)
    df = load_raw()
    df = run_cleaning(df)
    _train_df, _val_df, test_df = make_splits(df)
    logger.info("Jeu de test : %d observations", len(test_df))

    # ── Etape 3 : DataLoader de test ─────────────────────────────────────────
    logger.info("Etape 3/4 : Creation du DataLoader")
    test_dataset = ClaimsDataset(test_df, vocab, label_encoder)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # ── Etape 4 : evaluation ─────────────────────────────────────────────────
    logger.info("Etape 4/4 : Evaluation en cours...")
    results = evaluate(model_obj, test_loader, label_encoder, device)

    results.print_summary(f"Test — {arch_name.upper()}")
    check_objective(results)

    # ── Livrables ────────────────────────────────────────────────────────────
    cm_path = plot_confusion_matrix(
        results,
        class_names=label_encoder.classes,
        model_name=arch_name,
    )
    report_path = save_text_report(
        results,
        model_name=arch_name,
        extra_info=f"Checkpoint : best_val_f1={best_val_f1:.4f}\n",
    )

    logger.info("Matrice de confusion : %s", cm_path)
    logger.info("Rapport texte        : %s", report_path)


if __name__ == "__main__":
    app()
