# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Calcul et chargement de la baseline de reference.

La baseline est calculee UNE SEULE FOIS a partir du jeu d'entrainement.
Elle capture :
  - La distribution des classes attendue (proportions αi)
  - La distribution des longueurs de texte (statistiques descriptives)
  - Le taux moyen de tokens inconnus

Ces statistiques servent de reference pour la detection de derive en production.

Utilisation :
    from claims_classifier.monitoring.baseline import compute_baseline, load_baseline
    stats = load_baseline()   # charge depuis data/processed/baseline_stats.json
"""

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from claims_classifier.config import config
from claims_classifier.data.vocab import Vocabulary

logger = logging.getLogger(__name__)


def compute_baseline(train_df: pd.DataFrame, vocab: Vocabulary) -> dict:
    """
    Calcule les statistiques de reference a partir du jeu d'entrainement.

    Args:
        train_df : DataFrame d'entrainement (colonnes "text" et "label", deja nettoye).
        vocab    : Vocabulaire construit sur le train (pour calculer le taux d'<unk>).

    Returns:
        Dictionnaire de statistiques sauvegarde dans baseline_stats.json.
    """
    logger.info(f"Calcul de la baseline sur {len(train_df):,} observations...")

    # ── 1. Distribution des classes ───────────────────────────────────────────
    class_counts = train_df["label"].value_counts()
    total = len(train_df)
    class_distribution = {cls: round(count / total, 6) for cls, count in class_counts.items()}

    # ── 2. Longueur des textes (en mots, apres nettoyage) ────────────────────
    word_counts = train_df["text"].astype(str).str.split().str.len()
    text_length_stats = {
        "mean": round(float(word_counts.mean()), 2),
        "std": round(float(word_counts.std()), 2),
        "median": round(float(word_counts.median()), 2),
        "p25": round(float(word_counts.quantile(0.25)), 2),
        "p75": round(float(word_counts.quantile(0.75)), 2),
        "p95": round(float(word_counts.quantile(0.95)), 2),
        "min": int(word_counts.min()),
        "max": int(word_counts.max()),
    }

    # ── 3. Taux de tokens inconnus ───────────────────────────────────────────
    logger.info("Calcul du taux de tokens inconnus (peut prendre quelques secondes)...")

    def unk_rate_for_text(text: str) -> float:
        words = str(text).split()
        if not words:
            return 0.0
        n_unk = sum(1 for w in words if w not in vocab.word2idx)
        return n_unk / len(words)

    # Calcul sur un echantillon pour la rapidite (max 50 000 obs)
    sample = (
        train_df["text"]
        if len(train_df) <= 50_000
        else train_df["text"].sample(50_000, random_state=42)
    )
    unk_rates = sample.map(unk_rate_for_text)

    unk_rate_stats = {
        "mean": round(float(unk_rates.mean()), 6),
        "median": round(float(unk_rates.median()), 6),
        "p75": round(float(unk_rates.quantile(0.75)), 6),
        "p95": round(float(unk_rates.quantile(0.95)), 6),
    }

    # ── Assemblage ───────────────────────────────────────────────────────────
    baseline = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "total_train_samples": int(total),
        "num_classes": len(class_distribution),
        "class_distribution": class_distribution,
        "text_length_stats": text_length_stats,
        "unk_rate_stats": unk_rate_stats,
    }

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    out_path = config.monitoring.baseline_stats_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2, ensure_ascii=False)

    logger.info(f"Baseline sauvegardee : {out_path}")
    logger.info(f"  Classes        : {len(class_distribution)}")
    logger.info(
        f"  Longueur mots  : median={text_length_stats['median']:.0f}, p95={text_length_stats['p95']:.0f}"
    )
    logger.info(f"  Taux <unk>     : mean={unk_rate_stats['mean']:.4f}")

    return baseline


def load_baseline() -> dict | None:
    """
    Charge la baseline depuis baseline_stats.json.

    Returns:
        Dictionnaire de statistiques, ou None si le fichier n'existe pas.
    """
    path = config.monitoring.baseline_stats_path
    if not path.exists():
        logger.warning(f"Baseline introuvable : {path}. Lancez compute_baseline.py.")
        return None

    with open(path, encoding="utf-8") as fh:
        baseline = json.load(fh)

    logger.info(f"Baseline chargee : {path} (calculee le {baseline.get('computed_at', '?')})")
    return baseline
