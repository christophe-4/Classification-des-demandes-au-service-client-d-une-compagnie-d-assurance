# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Configuration centrale du projet claims-classifier.

"""

from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# CHEMINS DU PROJET
# =============================================================================

# Racine du projet absolue du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataPaths(BaseModel):
    """Chemins vers les donnees, organises en pipeline raw -> interim -> processed."""

    project_root: Path = PROJECT_ROOT

    # Repertoires
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # Fichiers
    raw_csv_path: Path = PROJECT_ROOT / "data" / "raw" / "complaints.csv"
    cleaned_csv_path: Path = PROJECT_ROOT / "data" / "interim" / "complaints_cleaned.csv"

    train_path: Path = PROJECT_ROOT / "data" / "processed" / "train.parquet"
    val_path: Path = PROJECT_ROOT / "data" / "processed" / "val.parquet"
    test_path: Path = PROJECT_ROOT / "data" / "processed" / "test.parquet"

    vocab_path: Path = PROJECT_ROOT / "data" / "processed" / "vocab.json"
    label_encoder_path: Path = PROJECT_ROOT / "data" / "processed" / "label_encoder.json"


class ArtifactsPaths(BaseModel):
    """Chemins vers les artefacts produits (modeles, logs, figures)."""

    models_dir: Path = PROJECT_ROOT / "models"
    runs_dir: Path = PROJECT_ROOT / "runs"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"
    reports_dir: Path = PROJECT_ROOT / "reports"


# =============================================================================
# HYPERPARAMETRES
# =============================================================================

class PreprocessingConfig(BaseModel):
    """Parametres de pretraitement du texte."""

    money_token: str = "<money>"
    date_token: str = "<date>"

    # Longueur max d'une sequence (en tokens)
    # p50=118, p75=215, p90=377, p95=540 sur 300k réclamations
    # 512 couvre ~85% du corpus avec marge raisonnable
    max_seq_length: int = 256

    # Vocabulaire
    # Mots uniques bruts : 193k, vus >= 2 fois : 89k, >= 5 fois : 48k
    # 30k capture les mots fréquents tout en filtrant bruit et hapax
    vocab_size: int = 30_000
    min_word_frequency: int = 2  # Filtre 193k -> 89k mots uniques utiles

    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    pad_token_id: int = 0
    unk_token_id: int = 1


class SplitConfig(BaseModel):
    """Parametres de decoupage train/val/test."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = True
    seed: int = 42


class ModelConfig(BaseModel):
    """Hyperparametres des modeles."""

    # Embedding (commun aux deux modeles)
    embedding_dim: int = 128

    # MLP
    mlp_hidden_dim: int = 64
    mlp_dropout: float = 0.3

    # TextCNN
    cnn_num_filters: int = 128
    cnn_kernel_sizes: tuple[int, ...] = (3, 4, 5)
    cnn_dropout: float = 0.5


class TrainingConfig(BaseModel):
    """Parametres d'entrainement."""

    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5  # Regularisation L2 (cours)
    early_stopping_patience: int = 3
    gradient_clip_value: float = 1.0  # Stabilite numerique (cours)

    # Reproductibilite
    seed: int = 42

    # Materiel
    num_workers: int = 0  # 0 pour Windows, 4 pour Linux
    device: str = "cuda"  # Sera bascule en "cpu" automatiquement si pas de GPU


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

class Config(BaseSettings):
    """
    Configuration complete du projet.

    Peut etre surchargee par variables d'environnement, par exemple :
        export CLAIMS_TRAINING__BATCH_SIZE=128
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAIMS_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    data: DataPaths = Field(default_factory=DataPaths)
    artifacts: ArtifactsPaths = Field(default_factory=ArtifactsPaths)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# Instance globale, importable partout dans le projet
config = Config()


# =============================================================================
# UTILITAIRE
# =============================================================================

def print_config() -> None:
    """Affiche la configuration courante (utile pour debugger)."""
    import json
    print(json.dumps(config.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    print_config()