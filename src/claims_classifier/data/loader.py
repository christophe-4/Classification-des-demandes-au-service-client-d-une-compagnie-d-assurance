"""
Chargement des donnees brutes.

Responsabilite unique : lire complaints.csv et retourner un DataFrame propre.
Le nettoyage du texte est dans cleaning.py.
"""

import logging
from pathlib import Path

import pandas as pd

from claims_classifier.config import config

logger = logging.getLogger(__name__)


# Colonnes attendues dans le fichier brut
COL_TEXT = "Consumer complaint narrative"
COL_LABEL = "Product"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """
    Charge le fichier CSV brut et retourne un DataFrame minimal.

    Args:
        path: Chemin vers le CSV. Si None, utilise config.data.raw_csv_path.

    Returns:
        DataFrame avec deux colonnes : 'text' et 'label'.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si les colonnes attendues sont absentes.
    """
    csv_path = path or config.data.raw_csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")

    logger.info(f"Chargement de : {csv_path}")

    df = pd.read_csv(
        csv_path,
        usecols=[COL_LABEL, COL_TEXT],  # on ne charge que ce dont on a besoin
        dtype={COL_LABEL: "string", COL_TEXT: "string"},
        engine="python",  # gere les textes multi-lignes avec guillemets
    )

    # Verifier que les colonnes attendues sont presentes
    missing = {COL_LABEL, COL_TEXT} - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    # Renommer pour simplifier — noms courts utilises partout dans le projet
    df = df.rename(columns={COL_TEXT: "text", COL_LABEL: "label"})

    # Retirer les lignes sans texte ou sans label
    n_before = len(df)
    df = df.dropna(subset=["text", "label"])
    n_after = len(df)

    if n_before != n_after:
        logger.warning(f"{n_before - n_after} lignes supprimees (texte ou label manquant)")

    df = df.reset_index(drop=True)

    logger.info(f"Dataset charge : {len(df):,} observations, {df['label'].nunique()} classes")

    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """
    Affiche un resume du dataset dans le terminal.
    Utile pour une verification rapide avant l'EDA complete.
    """
    print(f"\n{'=' * 50}")
    print("RESUME DU DATASET")
    print(f"{'=' * 50}")
    print(f"Observations totales : {len(df):,}")
    print(f"Nombre de classes    : {df['label'].nunique()}")
    print("\nDistribution des classes (n_i) :")
    print(f"{'-' * 50}")

    counts = df["label"].value_counts()
    total = len(df)

    for label, count in counts.items():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {label[:45]:<45} {count:>6,} ({pct:5.1f}%) {bar}")

    print(f"{'-' * 50}")
    print("\nLongueur des textes (en caracteres) :")
    lengths = df["text"].str.len()
    print(f"  Min    : {lengths.min():>8,}")
    print(f"  Median : {lengths.median():>8,.0f}")
    print(f"  Mean   : {lengths.mean():>8,.0f}")
    print(f"  p75    : {lengths.quantile(0.75):>8,.0f}")
    print(f"  p95    : {lengths.quantile(0.95):>8,.0f}")
    print(f"  Max    : {lengths.max():>8,}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    df = load_raw()
    describe_dataset(df)
