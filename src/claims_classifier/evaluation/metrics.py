"""
Metriques d'evaluation du modele de classification.

Metriques implementees :
  - Weighted F1 Score
  - Macro F1 Score
  - F1 par classe
  - Accuracy

"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader

from claims_classifier.data.dataset import LabelEncoder

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASS DE RESULTATS
# =============================================================================


@dataclass
class EvaluationResults:
    """
    Conteneur pour tous les resultats d'evaluation.

    Attributes:
        weighted_f1  : Metrique principale (brief : doit etre >= 75%).
        macro_f1     : F1 moyen non pondere (verifie qu'aucune classe n'est sacrifiee).
        accuracy     : Proportion de predictions correctes.
        f1_per_class : F1 individuel pour chaque classe.
        report       : Rapport complet (scikit-learn classification_report).
        all_preds    : Toutes les predictions (pour la matrice de confusion).
        all_labels   : Tous les vrais labels (pour la matrice de confusion).
    """

    weighted_f1: float
    macro_f1: float
    accuracy: float
    f1_per_class: dict[str, float]
    report: str
    all_preds: list[int] = field(default_factory=list)
    all_labels: list[int] = field(default_factory=list)

    def print_summary(self, title: str = "Resultats") -> None:
        """Affiche un resume lisible des metriques principales."""
        target = 0.75
        status = "OBJECTIF ATTEINT" if self.weighted_f1 >= target else "OBJECTIF NON ATTEINT"

        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(f"  Weighted F1  : {self.weighted_f1:.4f}  ({self.weighted_f1 * 100:.2f}%)  {status}")
        print(f"  Macro F1     : {self.macro_f1:.4f}  ({self.macro_f1 * 100:.2f}%)")
        print(f"  Accuracy     : {self.accuracy:.4f}  ({self.accuracy * 100:.2f}%)")
        print("\n  F1 par classe :")
        print(f"  {'-' * 45}")
        for cls, f1 in sorted(self.f1_per_class.items(), key=lambda x: x[1]):
            bar = "#" * int(f1 * 20)
            flag = " (!)" if f1 < 0.5 else ""
            print(f"  {cls:<30} {f1:.4f}  {bar}{flag}")
        print(f"{'=' * 60}\n")


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> EvaluationResults:
    """
    Evalue le modele sur un DataLoader et retourne toutes les metriques.

    Args:
        model         : Modele entraine (MLP ou TextCNN).
        loader        : DataLoader (val ou test).
        label_encoder : Pour decoder les IDs en noms de classes.
        device        : Dispositif de calcul.

    Returns:
        EvaluationResults avec toutes les metriques calculees.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    class_names = label_encoder.classes
    # Liste explicite des indices de classes (0..num_classes-1)
    # Garantit l'alignement scores <-> classes meme si une classe est absente
    class_indices = list(range(label_encoder.num_classes))

    # Metriques globales
    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        labels=class_indices,
        average="weighted",
        zero_division=0,
    )
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=class_indices,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(all_labels, all_preds)

    # F1 par classe — l'ordre suit class_indices, donc class_names
    f1_scores = f1_score(
        all_labels,
        all_preds,
        labels=class_indices,
        average=None,
        zero_division=0,
    )
    f1_per_class = {class_names[i]: float(f1_scores[i]) for i in range(len(class_names))}

    # Rapport complet scikit-learn
    report = classification_report(
        all_labels,
        all_preds,
        labels=class_indices,
        target_names=class_names,
        zero_division=0,
    )

    return EvaluationResults(
        weighted_f1=weighted_f1,
        macro_f1=macro_f1,
        accuracy=accuracy,
        f1_per_class=f1_per_class,
        report=report,
        all_preds=all_preds,
        all_labels=all_labels,
    )


def check_objective(results: EvaluationResults) -> bool:
    """
    Verifie si l'objectif est atteint (Weighted F1 >= 75%).

    Args:
        results: Resultats d'evaluation.

    Returns:
        True si l'objectif est atteint, False sinon.
    """
    target = 0.75
    passed = results.weighted_f1 >= target

    if passed:
        logger.info(f"Objectif atteint : Weighted F1 = {results.weighted_f1:.4f} >= {target}")
    else:
        logger.warning(f"Objectif non atteint : Weighted F1 = {results.weighted_f1:.4f} < {target}")

    return passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    # Test avec des predictions aleatoires pour verifier la structure
    import random

    class_names = [
        "bank_account_or_service",
        "checking_or_savings",
        "consumer_loan",
        "credit_card",
        "credit_reporting",
        "debt_collection",
        "money_transfer",
        "mortgage",
        "other",
        "payday_loan",
        "student_loan",
        "vehicle_loan",
    ]

    n = 1000
    all_labels = [random.randint(0, 11) for _ in range(n)]
    all_preds = [random.randint(0, 11) for _ in range(n)]

    num_classes = len(class_names)
    class_indices = list(range(num_classes))

    weighted_f1 = f1_score(
        all_labels, all_preds, labels=class_indices, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(
        all_labels, all_preds, labels=class_indices, average="macro", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    f1_scores = f1_score(all_labels, all_preds, labels=class_indices, average=None, zero_division=0)
    f1_per_class = {class_names[i]: float(f1_scores[i]) for i in range(num_classes)}
    report = classification_report(
        all_labels, all_preds, labels=class_indices, target_names=class_names, zero_division=0
    )

    results = EvaluationResults(
        weighted_f1=weighted_f1,
        macro_f1=macro_f1,
        accuracy=accuracy,
        f1_per_class=f1_per_class,
        report=report,
        all_preds=all_preds,
        all_labels=all_labels,
    )

    results.print_summary("Test predictions aleatoires (reference = ~8%)")
    check_objective(results)
    print("\nClassification report :")
    print(report)
    print(" metrics.py operationnel")
