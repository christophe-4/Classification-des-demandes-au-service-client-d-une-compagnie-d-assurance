"""
Dataset PyTorch et utilitaires de splitting/chargement.

Responsabilite :
  - ClaimsDataset : interface PyTorch pour les reclamations
                    (padding a max_seq_length effectue dans __getitem__,
                     donc pas besoin de collate_fn custom)
  - make_splits   : decoupage stratifie train/val/test
  - make_loaders  : creation des DataLoaders prets a l'emploi
"""

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from claims_classifier.config import config
from claims_classifier.data.vocab import Vocabulary

logger = logging.getLogger(__name__)


# =============================================================================
# LABEL ENCODER
# =============================================================================


class LabelEncoder:
    """
    Encode les labels textuels en entiers et vice-versa.

    Exemple :
        encoder = LabelEncoder.build(df["label"])
        encoder.encode("credit_reporting")  # -> 0
        encoder.decode(0)                   # -> "credit_reporting"
    """

    def __init__(self) -> None:
        self.label2idx: dict[str, int] = {}
        self.idx2label: dict[int, str] = {}

    @classmethod
    def build(cls, labels: pd.Series) -> "LabelEncoder":
        """Construit l'encodeur depuis une serie de labels."""
        encoder = cls()
        unique_labels = sorted(labels.unique())
        encoder.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        encoder.idx2label = {idx: label for label, idx in encoder.label2idx.items()}
        logger.info(f"LabelEncoder construit : {len(encoder.label2idx)} classes")
        return encoder

    def encode(self, label: str) -> int:
        if label not in self.label2idx:
            raise ValueError(f"Label inconnu : '{label}'")
        return self.label2idx[label]

    def decode(self, idx: int) -> str:
        if idx not in self.idx2label:
            raise ValueError(f"Index inconnu : {idx}")
        return self.idx2label[idx]

    @property
    def num_classes(self) -> int:
        return len(self.label2idx)

    @property
    def classes(self) -> list[str]:
        """Liste des classes dans l'ordre de leurs IDs."""
        return [self.idx2label[i] for i in range(self.num_classes)]

    def save(self, path: Path | None = None) -> None:
        save_path = path or config.data.label_encoder_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.label2idx, f, ensure_ascii=False, indent=2)
        logger.info(f"LabelEncoder sauvegarde : {save_path}")

    @classmethod
    def load(cls, path: Path | None = None) -> "LabelEncoder":
        load_path = path or config.data.label_encoder_path
        if not load_path.exists():
            raise FileNotFoundError(f"LabelEncoder introuvable : {load_path}")
        encoder = cls()
        with open(load_path, "r", encoding="utf-8") as f:
            encoder.label2idx = json.load(f)
        encoder.idx2label = {idx: label for label, idx in encoder.label2idx.items()}
        logger.info(f"LabelEncoder charge : {encoder.num_classes} classes")
        return encoder


# =============================================================================
# DATASET PYTORCH
# =============================================================================


class ClaimsDataset(Dataset):
    """
    Dataset PyTorch pour les reclamations clients.

    Chaque element retourne :
      - input_ids (LongTensor) : sequence de token IDs, paddee a max_seq_length
      - label (LongTensor)     : ID de la classe (scalaire)

    Exemple :
        dataset = ClaimsDataset(df, vocab, encoder)
        input_ids, label = dataset[0]
        print(input_ids.shape)  # torch.Size([256])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        label_encoder: LabelEncoder,
        max_seq_length: int | None = None,
    ) -> None:
        """
        Args:
            df            : DataFrame avec colonnes 'text' et 'label' (nettoyes).
            vocab         : Vocabulaire construit sur le train.
            label_encoder : Encodeur de labels construit sur le train.
            max_seq_length: Longueur max des sequences (defaut : config).
        """
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_seq_length = (
            max_seq_length if max_seq_length is not None else config.preprocessing.max_seq_length
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Texte -> IDs (tronque si trop long)
        ids = self.vocab.encode(text)[: self.max_seq_length]

        # Padding a droite si trop court
        pad_length = self.max_seq_length - len(ids)
        ids = ids + [self.vocab.pad_id] * pad_length

        input_ids = torch.tensor(ids, dtype=torch.long)
        label_id = torch.tensor(self.label_encoder.encode(label), dtype=torch.long)

        return input_ids, label_id


# =============================================================================
# SPLITTING STRATIFIE
# =============================================================================


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Decoupe le DataFrame en train / val / test de facon stratifiee.

    Stratifie sur 'label' : chaque split a la meme distribution de classes.
    Utilise la seed de config pour la reproductibilite.

    Args:
        df: DataFrame complet (nettoye).

    Returns:
        Tuple (train_df, val_df, test_df).
    """
    seed = config.split.seed
    val_ratio = config.split.val_ratio
    test_ratio = config.split.test_ratio

    use_stratify = config.split.stratify

    # Etape 1 : separer test du reste
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label"] if use_stratify else None,
        random_state=seed,
    )

    # Etape 2 : separer val du train_val
    # val_ratio ajuste par rapport a la taille de train_val
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df["label"] if use_stratify else None,
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Splits : train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")

    return train_df, val_df, test_df


# =============================================================================
# DATALOADERS
# =============================================================================


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab: Vocabulary,
    label_encoder: LabelEncoder,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cree les DataLoaders PyTorch pour train, val et test.

    Le DataLoader gere :
      - Le decoupage en batches (batch_size depuis config)
      - Le melange aleatoire (shuffle=True sur train seulement)
      - Le parallelisme de chargement (num_workers depuis config)

    Args:
        train_df, val_df, test_df : DataFrames des trois splits.
        vocab         : Vocabulaire.
        label_encoder : Encodeur de labels.

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    batch_size = config.training.batch_size
    num_workers = config.training.num_workers

    train_dataset = ClaimsDataset(train_df, vocab, label_encoder)
    val_dataset = ClaimsDataset(val_df, vocab, label_encoder)
    test_dataset = ClaimsDataset(test_df, vocab, label_encoder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # melange a chaque epoque (cours : SGD stochastique)
        num_workers=num_workers,
        pin_memory=True,  # accelere le transfert CPU -> GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders prets — "
        f"train: {len(train_loader)} batches | "
        f"val: {len(val_loader)} batches | "
        f"test: {len(test_loader)} batches "
        f"(batch_size={batch_size})"
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# POINT D'ENTREE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    from claims_classifier.data.cleaning import run_cleaning
    from claims_classifier.data.loader import load_raw

    # Pipeline complet
    df = load_raw()
    df = run_cleaning(df)

    # Splits stratifies
    train_df, val_df, test_df = make_splits(df)

    # Vocabulaire et encodeur construits sur le TRAIN uniquement
    # (regle fondamentale : pas de fuite de donnees du val/test vers le train)
    vocab = Vocabulary.build(train_df["text"])
    vocab.save()

    label_encoder = LabelEncoder.build(train_df["label"])
    label_encoder.save()

    # DataLoaders
    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, vocab, label_encoder
    )

    # Verification : inspecter un batch
    print("\n--- Verification d'un batch ---")
    input_ids, labels = next(iter(train_loader))
    print(f"input_ids shape : {input_ids.shape}")  # [64, 256]
    print(f"labels shape    : {labels.shape}")  # [64]
    print(f"input_ids dtype : {input_ids.dtype}")  # torch.int64
    print(f"labels dtype    : {labels.dtype}")  # torch.int64
    print("\nPremier exemple du batch :")
    print(f"  IDs     : {input_ids[0][:10].tolist()}... (256 total)")
    print(f"  Label   : {labels[0].item()} -> '{label_encoder.decode(labels[0].item())}'")
    print(f"\nClasses disponibles ({label_encoder.num_classes}) :")
    for idx, cls in enumerate(label_encoder.classes):
        print(f"  {idx:2d} -> {cls}")
