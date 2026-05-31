# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Script de calcul de la baseline de reference.

Necessite :
  - data/raw/complaints.csv     (dataset CFPB)
  - data/processed/vocab.json   (vocabulaire construit lors de l'entrainement)

Produit :
  - data/processed/baseline_stats.json

Usage :
    uv run python scripts/compute_baseline.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claims_classifier.config import config
from claims_classifier.data.cleaning import run_cleaning
from claims_classifier.data.dataset import make_splits
from claims_classifier.data.loader import load_raw
from claims_classifier.data.vocab import Vocabulary
from claims_classifier.monitoring.baseline import compute_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Pipeline complet de calcul de la baseline."""
    # ── Verifications prealables ──────────────────────────────────────────────
    if not config.data.raw_csv_path.exists():
        logger.error(f"Dataset introuvable : {config.data.raw_csv_path}")
        logger.error("Placez complaints.csv dans data/raw/ avant de continuer.")
        sys.exit(1)

    if not config.data.vocab_path.exists():
        logger.error(f"Vocabulaire introuvable : {config.data.vocab_path}")
        logger.error("Lancez d'abord l'entrainement : uv run python scripts/train.py")
        sys.exit(1)

    # ── Pipeline data ─────────────────────────────────────────────────────────
    logger.info("Chargement du dataset brut...")
    df = load_raw()

    logger.info("Nettoyage...")
    df = run_cleaning(df)

    logger.info("Split train/val/test (seed=42, identique a l'entrainement)...")
    train_df, _, _ = make_splits(df)
    logger.info(f"Jeu d'entrainement : {len(train_df):,} observations")

    # ── Chargement du vocabulaire ─────────────────────────────────────────────
    logger.info("Chargement du vocabulaire...")
    vocab = Vocabulary.load()

    # ── Calcul et sauvegarde ──────────────────────────────────────────────────
    stats = compute_baseline(train_df, vocab)

    print("\n" + "=" * 60)
    print("BASELINE CALCULEE")
    print("=" * 60)
    print(f"  Fichier         : {config.monitoring.baseline_stats_path}")
    print(f"  Observations    : {stats['total_train_samples']:,}")
    print(f"  Classes         : {stats['num_classes']}")
    print(f"  Longueur mediane: {stats['text_length_stats']['median']:.0f} mots")
    print(f"  Taux <unk> moyen: {stats['unk_rate_stats']['mean']:.4f}")
    print("\nDistribution des classes :")
    for cls, prop in sorted(stats["class_distribution"].items(), key=lambda x: -x[1]):
        bar = "#" * int(prop * 40)
        print(f"  {cls:<35} {prop * 100:5.1f}%  {bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()
