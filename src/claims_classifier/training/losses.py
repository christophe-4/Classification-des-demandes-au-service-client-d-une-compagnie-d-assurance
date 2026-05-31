"""
Fonction de perte ponderee pour la classification desequilibree.

Probleme : ratio max/min = 957:1 entre les classes.
Solution  : CrossEntropyLoss avec poids inversement proportionnels
            a la frequence de chaque classe (cours : fonction objectif).

Formule du poids :
    weight[i] = n_total / (n_classes * n_i)

Effet :
    - Classe rare  (n_i faible) -> poids eleve  -> le modele penalise fort les erreurs
    - Classe dominante (n_i eleve) -> poids faible -> moins de penalite

Reference cours : "Optimisation numerique" + "Regularisation et stabilite"
"""

import logging

import pandas as pd
import torch
import torch.nn as nn

from claims_classifier.data.dataset import LabelEncoder

logger = logging.getLogger(__name__)


def compute_class_weights(
    train_labels: pd.Series,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> torch.Tensor:
    """
    Calcule les poids de chaque classe pour la CrossEntropyLoss.

    Formule : weight[i] = n_total / (n_classes * n_i)

    Args:
        train_labels  : Serie pandas des labels du train (apres encodage textuel).
        label_encoder : Pour recuperer l'ordre des classes.
        device        : Dispositif cible (cpu ou cuda).

    Returns:
        Tenseur de poids [num_classes] sur le bon device.
    """
    n_total = len(train_labels)
    n_classes = label_encoder.num_classes
    counts = train_labels.value_counts()

    weights = []
    logger.info("Poids des classes (CrossEntropy ponderee) :")
    logger.info(f"  {'Classe':<30} {'n_i':>8} {'weight':>8}")
    logger.info(f"  {'-' * 50}")

    for class_name in label_encoder.classes:
        n_i = counts.get(class_name, 1)
        weight = n_total / (n_classes * n_i)
        weights.append(weight)
        logger.info(f"  {class_name:<30} {n_i:>8,} {weight:>8.4f}")

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return weight_tensor


def build_loss(
    train_labels: pd.Series,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> nn.CrossEntropyLoss:
    """
    Construit la fonction de perte CrossEntropy ponderee.

    nn.CrossEntropyLoss combine en interne :
      1. LogSoftmax  : normalise les logits en log-probabilites
      2. NLLLoss     : negative log-likelihood

    C'est plus stable numeriquement que d'appliquer Softmax puis log
    separement

    Args:
        train_labels  : Labels du jeu d'entrainement.
        label_encoder : Encodeur de labels.
        device        : Dispositif cible.

    Returns:
        Fonction de perte configuree et prete a l'emploi.
    """
    class_weights = compute_class_weights(train_labels, label_encoder, device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    logger.info(
        f"Loss : CrossEntropyLoss ponderee | "
        f"min_weight={class_weights.min():.4f} | "
        f"max_weight={class_weights.max():.4f}"
    )

    return loss_fn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    from claims_classifier.data.cleaning import run_cleaning
    from claims_classifier.data.dataset import LabelEncoder, make_splits
    from claims_classifier.data.loader import load_raw

    df = load_raw()
    df = run_cleaning(df)
    train_df, val_df, test_df = make_splits(df)

    label_encoder = LabelEncoder.build(train_df["label"])
    device = torch.device("cpu")

    loss_fn = build_loss(train_df["label"], label_encoder, device)

    # Verification : loss sur un batch fictif
    batch_size = 4
    num_classes = label_encoder.num_classes
    fake_logits = torch.randn(batch_size, num_classes)
    fake_labels = torch.randint(0, num_classes, (batch_size,))

    loss_value = loss_fn(fake_logits, fake_labels)

    print("\nVerification :")
    print(f"  Logits : {fake_logits.shape}")
    print(f"  Labels : {fake_labels.tolist()}")
    print(f"  Loss   : {loss_value.item():.4f}")
    print("\nLoss ponderee operationnelle")
