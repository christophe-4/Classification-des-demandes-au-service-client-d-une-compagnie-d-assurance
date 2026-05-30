"""
Construction et gestion du vocabulaire.

Responsabilite : transformer les mots en identifiants numeriques (token IDs)
et vice-versa. Le vocabulaire est construit sur le corpus d'entrainement
uniquement.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from claims_classifier.config import config

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Vocabulaire mot -> ID et ID -> mot.

    Tokens speciaux reserves :
      <pad> (ID=0) : rembourrage des sequences courtes
      <unk> (ID=1) : mots absents du vocabulaire

    Exemple d'utilisation :
        vocab = Vocabulary.build(texts)
        ids = vocab.encode("credit report issue")  # [42, 17, 305]
        words = vocab.decode([42, 17, 305])         # ["credit", "report", "issue"]
    """

    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._size: int = 0

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        texts: pd.Series,
        vocab_size: int | None = None,
        min_frequency: int | None = None,
    ) -> "Vocabulary":
        """
        Construit le vocabulaire a partir d'une serie de textes.

        Args:
            texts: Serie pandas de textes (deja nettoyes).
            vocab_size: Nombre maximum de mots (hors tokens speciaux).
                        Defaut : config.preprocessing.vocab_size.
            min_frequency: Frequence minimale pour inclure un mot.
                           Defaut : config.preprocessing.min_word_frequency.

        Returns:
            Instance de Vocabulary prete a l'emploi.
        """
        vocab_size = vocab_size if vocab_size is not None else config.preprocessing.vocab_size
        min_frequency = min_frequency if min_frequency is not None else config.preprocessing.min_word_frequency

        logger.info("Construction du vocabulaire...")

        # Compter les occurrences de chaque mot
        counter: Counter = Counter()
        for text in texts:
            counter.update(str(text).split())

        logger.info(f"Mots uniques dans le corpus : {len(counter):,}")

        # Filtrer par frequence minimale
        counter = Counter({w: c for w, c in counter.items() if c >= min_frequency})
        logger.info(f"Apres filtrage (freq >= {min_frequency}) : {len(counter):,} mots")

        # Garder les N mots les plus frequents
        most_common = counter.most_common(vocab_size)

        vocab = cls()
        vocab._build_from_words([w for w, _ in most_common])

        logger.info(
            f"Vocabulaire construit : {vocab.size:,} mots "
            f"(+ 2 tokens speciaux <pad>, <unk>)"
        )

        return vocab

    def _build_from_words(self, words: list[str]) -> None:
        """Construit les tables word2idx et idx2word a partir d'une liste de mots."""
        # Tokens speciaux en premier (IDs fixes)
        self.word2idx = {
            config.preprocessing.pad_token: config.preprocessing.pad_token_id,
            config.preprocessing.unk_token: config.preprocessing.unk_token_id,
        }

        # Mots du corpus a partir de l'ID 2
        for idx, word in enumerate(words, start=2):
            self.word2idx[word] = idx

        # Table inverse
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self._size = len(self.word2idx)

    # -------------------------------------------------------------------------
    # Encodage / Decodage
    # -------------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Convertit un texte en liste d'IDs.

        Les mots absents du vocabulaire sont remplaces par <unk> (ID=1).

        Args:
            text: Texte pre-nettoye (minuscules, sans ponctuation).

        Returns:
            Liste d'entiers (token IDs).
        """
        unk_id = config.preprocessing.unk_token_id
        return [self.word2idx.get(word, unk_id) for word in text.split()]

    def decode(self, ids: list[int]) -> list[str]:
        """
        Convertit une liste d'IDs en mots.

        Args:
            ids: Liste d'entiers (token IDs).

        Returns:
            Liste de mots.
        """
        pad_id = config.preprocessing.pad_token_id
        return [self.idx2word.get(i, config.preprocessing.unk_token)
                for i in ids if i != pad_id]

    # -------------------------------------------------------------------------
    # Persistance
    # -------------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Sauvegarde le vocabulaire en JSON."""
        save_path = path or config.data.vocab_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

        logger.info(f"Vocabulaire sauvegarde : {save_path}")

    @classmethod
    def load(cls, path: Path | None = None) -> "Vocabulary":
        """Charge un vocabulaire depuis un fichier JSON."""
        load_path = path or config.data.vocab_path

        if not load_path.exists():
            raise FileNotFoundError(f"Vocabulaire introuvable : {load_path}")

        vocab = cls()

        with open(load_path, "r", encoding="utf-8") as f:
            vocab.word2idx = json.load(f)

        vocab.idx2word = {idx: word for word, idx in vocab.word2idx.items()}
        vocab._size = len(vocab.word2idx)

        logger.info(f"Vocabulaire charge : {vocab.size:,} mots depuis {load_path}")

        return vocab

    # -------------------------------------------------------------------------
    # Proprietes
    # -------------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Taille totale du vocabulaire (tokens speciaux inclus)."""
        return self._size

    @property
    def pad_id(self) -> int:
        return config.preprocessing.pad_token_id

    @property
    def unk_id(self) -> int:
        return config.preprocessing.unk_token_id

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size:,})"


# =============================================================================
# POINT D'ENTREE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    from claims_classifier.data.loader import load_raw
    from claims_classifier.data.cleaning import run_cleaning
    from claims_classifier.data.dataset import make_splits

    # Charger et nettoyer
    df = load_raw()
    df = run_cleaning(df)

    # Split stratifie : vocabulaire construit sur le TRAIN uniquement (anti-fuite)
    train_df, _val_df, _test_df = make_splits(df)

    vocab = Vocabulary.build(train_df["text"])

    # Verification
    print(f"\n{vocab}")
    print("\nExemple d'encodage :")
    sample = "credit report violation federal law <money>"
    ids = vocab.encode(sample)
    print(f"  Texte : {sample}")
    print(f"  IDs   : {ids}")
    print(f"  Retour: {vocab.decode(ids)}")

    # Sauvegarde
    vocab.save()
    print(f"\nVocabulaire sauvegarde dans : {config.data.vocab_path}")