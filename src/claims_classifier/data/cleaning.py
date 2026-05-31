"""
Nettoyage des donnees : mapping des labels (fusion) + nettoyage du texte.

Deux responsabilites :
  1. Regrouper les 21 classes brutes en 12 classes finales coherentes.
  2. Nettoyer le texte anonymise (XXXX, dates, montants, ponctuation).
"""

import logging
import re

import pandas as pd

from claims_classifier.config import config

logger = logging.getLogger(__name__)


# =============================================================================
# 1. MAPPING DES LABELS (fusion 21 -> 12 classes)
# =============================================================================

LABEL_MAPPING: dict[str, str] = {
    # --- Groupe 1 : Credit reporting (3 -> 1) ---
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
    "Credit reporting or other personal consumer reports": "credit_reporting",
    "Credit reporting": "credit_reporting",
    # --- Groupe 2 : Credit card (3 -> 1) ---
    "Credit card or prepaid card": "credit_card",
    "Credit card": "credit_card",
    "Prepaid card": "credit_card",
    # --- Groupe 3 : Money transfer (3 -> 1) ---
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Money transfers": "money_transfer",
    "Virtual currency": "money_transfer",
    # --- Groupe 4 : Payday loan (3 -> 1) ---
    "Payday loan, title loan, or personal loan": "payday_loan",
    "Payday loan, title loan, personal loan, or advance loan": "payday_loan",
    "Payday loan": "payday_loan",
    # --- Classes conservees telles quelles ---
    "Debt collection": "debt_collection",
    "Mortgage": "mortgage",
    "Checking or savings account": "checking_or_savings",
    "Student loan": "student_loan",
    "Vehicle loan or lease": "vehicle_loan",
    "Bank account or service": "bank_account_or_service",
    "Consumer Loan": "consumer_loan",
    # --- Classes ultra-rares -> fusionnees en "other" ---
    "Debt or credit management": "other",
    "Other financial service": "other",
}


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le mapping des labels (fusion des classes redondantes).

    Args:
        df: DataFrame avec une colonne 'label' contenant les libelles bruts.

    Returns:
        DataFrame avec 'label' remplace par les classes finales.
        Les libelles non references dans LABEL_MAPPING sont supprimes (avec warning).
    """
    df = df.copy()

    # Detecter les labels inconnus (non prevus dans le mapping)
    known_labels = set(LABEL_MAPPING.keys())
    actual_labels = set(df["label"].unique())
    unknown = actual_labels - known_labels

    if unknown:
        n_unknown = df["label"].isin(unknown).sum()
        logger.warning(f"{len(unknown)} libelles inconnus ({n_unknown} obs supprimees) : {unknown}")
        df = df[df["label"].isin(known_labels)]

    df["label"] = df["label"].map(LABEL_MAPPING)
    df = df.reset_index(drop=True)

    logger.info(f"Mapping applique : {df['label'].nunique()} classes finales")

    return df


# =============================================================================
# 2. NETTOYAGE DU TEXTE
# =============================================================================

# Patterns compiles une seule fois (performance sur 300k lignes)
RE_DATE = re.compile(r"\b(?:xx|x){1,2}/(?:xx|x){1,2}/(?:xx|x|\d){2,4}\b|\bxx/xx/year\b")
RE_MONEY = re.compile(r"\{?\$\s?[\d,]+\.?\d*\}?")
RE_XXXX = re.compile(r"\bx{2,}\b")
RE_NEWLINE = re.compile(r"\\n|\n|\r")
RE_NON_ALPHA = re.compile(r"[^a-z\s<>]")
RE_MULTISPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Nettoie un texte de reclamation.

    Pipeline :
      1. minuscules
      2. dates anonymisees -> <date>
      3. montants -> <money>
      4. sequences XXXX -> espace
      5. retours a la ligne -> espace
      6. ponctuation et caracteres speciaux -> espace (on garde < et > pour les tokens)
      7. espaces multiples -> espace simple

    Args:
        text: Texte brut.

    Returns:
        Texte nettoye.
    """
    text = text.lower()
    text = RE_NEWLINE.sub(" ", text)
    text = RE_DATE.sub(f" {config.preprocessing.date_token} ", text)
    text = RE_MONEY.sub(f" {config.preprocessing.money_token} ", text)
    text = RE_XXXX.sub(" ", text)
    text = RE_NON_ALPHA.sub(" ", text)
    text = RE_MULTISPACE.sub(" ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le nettoyage du texte sur tout le DataFrame.

    Args:
        df: DataFrame avec colonnes 'text' et 'label'.

    Returns:
        DataFrame nettoye. Les textes vides apres nettoyage sont supprimes.
    """
    df = df.copy()

    logger.info("Nettoyage du texte en cours...")
    df["text"] = df["text"].astype(str).map(clean_text)

    # Supprimer les textes devenus vides ou trop courts (< 3 mots)
    n_before = len(df)
    word_counts = df["text"].str.split().str.len()
    df = df[word_counts >= 3].reset_index(drop=True)
    n_after = len(df)

    if n_before != n_after:
        logger.warning(f"{n_before - n_after} textes supprimes (vides ou < 3 mots apres nettoyage)")

    logger.info(f"Nettoyage termine : {len(df):,} observations conservees")

    return df


# =============================================================================
# PIPELINE COMPLET
# =============================================================================


def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet : mapping des labels + nettoyage du texte.

    Args:
        df: DataFrame brut (sortie de loader.load_raw).

    Returns:
        DataFrame nettoye et pret pour la suite (vocab, dataset).
    """
    df = map_labels(df)
    df = clean_dataframe(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    from claims_classifier.data.loader import describe_dataset, load_raw

    df = load_raw()
    df = run_cleaning(df)
    describe_dataset(df)

    # Apercu de quelques textes nettoyes
    print("\n--- Apercu de 3 textes nettoyes ---")
    for i in range(3):
        print(f"\n[{df.loc[i, 'label']}]")
        print(df.loc[i, "text"][:300] + "...")
