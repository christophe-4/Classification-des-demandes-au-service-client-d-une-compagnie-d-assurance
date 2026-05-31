"""
Prediction de categorie pour un texte de reclamation libre.

Usage :
    uv run python scripts/predict.py --text "I have an issue with my credit report"
    uv run python scripts/predict.py --text "..." --model textcnn --top-k 5
    uv run python scripts/predict.py --text "..." --model mlp
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claims_classifier.config import config
from claims_classifier.data.cleaning import clean_text
from claims_classifier.inference.loader import load_for_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Prediction de categorie pour une reclamation client.")


@app.command()
def predict(
    text: str = typer.Option(
        ...,
        "--text",
        "-t",
        help="Texte de la reclamation a classifier.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Architecture : 'mlp' ou 'textcnn' (defaut : meilleur disponible).",
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        "-k",
        help="Nombre de predictions a afficher (defaut : 3).",
    ),
) -> None:
    """Classe un texte de reclamation et affiche les N meilleures predictions."""

    # ── Chargement du modele ─────────────────────────────────────────────────
    model_obj, vocab, label_encoder, arch_name, best_val_f1 = load_for_inference(model_name=model)
    device = next(model_obj.parameters()).device

    # ── Nettoyage du texte ───────────────────────────────────────────────────
    cleaned = clean_text(text)
    if not cleaned or len(cleaned.split()) < 1:
        typer.echo("Erreur : le texte est vide apres nettoyage.")
        raise typer.Exit(code=1)

    # ── Encodage + padding ───────────────────────────────────────────────────
    max_len = config.preprocessing.max_seq_length
    ids = vocab.encode(cleaned)[:max_len]
    ids = ids + [vocab.pad_id] * (max_len - len(ids))

    input_ids = torch.tensor([ids], dtype=torch.long).to(device)  # [1, max_len]

    # ── Inference ────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model_obj(input_ids)  # [1, num_classes]
        probs = torch.softmax(logits, dim=1)[0]  # [num_classes]

    # ── Top-k predictions ────────────────────────────────────────────────────
    k = min(top_k, label_encoder.num_classes)
    top_probs, top_indices = torch.topk(probs, k)

    # ── Affichage ────────────────────────────────────────────────────────────
    preview = text[:100] + ("..." if len(text) > 100 else "")
    cleaned_preview = cleaned[:100] + ("..." if len(cleaned) > 100 else "")

    print(f"\nTexte          : {preview}")
    print(f"Texte nettoye  : {cleaned_preview}")
    print(f"Modele         : {arch_name.upper()} (val F1={best_val_f1:.4f})")
    print(f"\nPredictions (top {k}) :")
    print(f"  {'Rang':<5} {'Categorie':<35} {'Probabilite':>12}")
    print("  " + "-" * 54)
    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        label = label_encoder.decode(idx)
        marker = "  <-- prediction" if rank == 1 else ""
        print(f"  {rank:<5} {label:<35} {prob * 100:>10.2f}%{marker}")
    print()


if __name__ == "__main__":
    app()
