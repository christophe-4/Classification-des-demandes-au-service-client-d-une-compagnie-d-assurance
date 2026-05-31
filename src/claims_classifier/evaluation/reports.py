"""
Generation des rapports visuels d'evaluation.

Livrables produits :
  - Matrice de confusion normalisee
  - Courbes d'entrainement (loss et F1 par epoque)
  - Rapport texte final
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from claims_classifier.config import config
from claims_classifier.evaluation.metrics import EvaluationResults

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    results: EvaluationResults,
    class_names: list[str],
    model_name: str = "model",
    normalize: bool = True,
) -> Path:
    """
    Genere et sauvegarde la matrice de confusion.

    La matrice est normalisee par ligne (valeurs entre 0 et 1) :
    chaque cellule indique la proportion de vrais labels de la ligne
    qui ont ete predits comme la colonne.
    La diagonale = taux de bonne classification par classe.

    Args:
        results     : Resultats d'evaluation (contient all_preds, all_labels).
        class_names : Noms des classes dans l'ordre des IDs.
        model_name  : Nom du modele (pour le nom du fichier).
        normalize   : Si True, normalise par ligne (defaut True).

    Returns:
        Chemin vers la figure sauvegardee.
    """
    # labels explicite : garantit que la matrice couvre toutes les classes
    # dans le bon ordre, meme si une classe est absente des predictions
    class_indices = list(range(len(class_names)))
    cm = confusion_matrix(results.all_labels, results.all_preds, labels=class_indices)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.where(
            row_sums == 0,
            0.0,
            cm.astype(float) / np.maximum(row_sums, 1),
        )
        fmt = ".2f"
        title = f"Matrice de confusion normalisee — {model_name.upper()}"
    else:
        cm_display = cm
        fmt = "d"
        title = f"Matrice de confusion — {model_name.upper()}"

    n_classes = len(class_names)
    fig_size = max(10, n_classes * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Classe predite", fontsize=12, labelpad=10)
    ax.set_ylabel("Vraie classe", fontsize=12, labelpad=10)
    ax.set_title(
        f"{title}\nWeighted F1 = {results.weighted_f1:.4f} "
        f"({'OK >= 75%' if results.weighted_f1 >= 0.75 else 'KO < 75%'})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    # Sauvegarde
    save_path = config.artifacts.figures_dir / f"confusion_matrix_{model_name}.png"
    config.artifacts.figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Matrice de confusion sauvegardee : {save_path}")
    return save_path


def plot_training_history(
    history: dict[str, list[float]],
    model_name: str = "model",
) -> Path:
    """
    Genere les courbes d'entrainement (loss et Weighted F1).

    Permet de visualiser :
      - La convergence du modele (loss qui diminue)
      - Le sur-apprentissage eventuel (ecart train/val, cours regularisation)
      - Le point d'arret de l'early stopping

    Args:
        history    : Dictionnaire retourne par Trainer.fit().
        model_name : Nom du modele.

    Returns:
        Chemin vers la figure sauvegardee.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Courbe Loss -----------------------------------------------------------
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=4, label="Validation")
    axes[0].set_xlabel("Epoque")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Fonction de perte — {model_name.upper()}", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -- Courbe Weighted F1 ----------------------------------------------------
    axes[1].plot(epochs, history["train_f1"], "b-o", markersize=4, label="Train")
    axes[1].plot(epochs, history["val_f1"], "r-o", markersize=4, label="Validation")
    axes[1].axhline(y=0.75, color="green", linestyle="--", linewidth=1.5, label="Objectif 75%")
    axes[1].set_xlabel("Epoque")
    axes[1].set_ylabel("Weighted F1")
    axes[1].set_title(f"Weighted F1 Score — {model_name.upper()}", fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Courbes d'entrainement — {model_name.upper()}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = config.artifacts.figures_dir / f"training_history_{model_name}.png"
    config.artifacts.figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Courbes d'entrainement sauvegardees : {save_path}")
    return save_path


def save_text_report(
    results: EvaluationResults,
    model_name: str = "model",
    extra_info: str = "",
) -> Path:
    """
    Sauvegarde le rapport textuel complet dans reports/.

    Args:
        results    : Resultats d'evaluation.
        model_name : Nom du modele.
        extra_info : Informations supplementaires a inclure.

    Returns:
        Chemin vers le rapport sauvegarde.
    """
    save_path = config.artifacts.reports_dir / f"report_{model_name}.txt"
    config.artifacts.reports_dir.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"RAPPORT D'EVALUATION — {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Weighted F1  : {results.weighted_f1:.4f} ({results.weighted_f1 * 100:.2f}%)\n")
        f.write(f"Macro F1     : {results.macro_f1:.4f} ({results.macro_f1 * 100:.2f}%)\n")
        f.write(f"Accuracy     : {results.accuracy:.4f} ({results.accuracy * 100:.2f}%)\n")

        status = "ATTEINT" if results.weighted_f1 >= 0.75 else "NON ATTEINT"
        f.write(f"Objectif 75% : {status}\n\n")

        f.write("F1 PAR CLASSE\n")
        f.write("-" * 40 + "\n")
        for cls, f1 in sorted(results.f1_per_class.items(), key=lambda x: x[1]):
            f.write(f"  {cls:<30} {f1:.4f}\n")

        f.write("\nCLASSIFICATION REPORT (scikit-learn)\n")
        f.write("-" * 40 + "\n")
        f.write(results.report)

        if extra_info:
            f.write("\nINFOS COMPLEMENTAIRES\n")
            f.write("-" * 40 + "\n")
            f.write(extra_info)

    logger.info(f"Rapport sauvegarde : {save_path}")
    return save_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    import random

    from sklearn.metrics import accuracy_score, classification_report, f1_score

    from claims_classifier.evaluation.metrics import EvaluationResults

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

    n = 500
    all_labels = [random.randint(0, 11) for _ in range(n)]
    all_preds = [random.randint(0, 11) for _ in range(n)]

    results = EvaluationResults(
        weighted_f1=f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        macro_f1=f1_score(all_labels, all_preds, average="macro", zero_division=0),
        accuracy=accuracy_score(all_labels, all_preds),
        f1_per_class={
            class_names[i]: float(s)
            for i, s in enumerate(f1_score(all_labels, all_preds, average=None, zero_division=0))
        },
        report=classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        ),
        all_preds=all_preds,
        all_labels=all_labels,
    )

    cm_path = plot_confusion_matrix(results, class_names, model_name="test")
    hist_path = plot_training_history(
        {
            "train_loss": [1.2, 1.0, 0.8],
            "val_loss": [1.3, 1.1, 0.9],
            "train_f1": [0.3, 0.5, 0.7],
            "val_f1": [0.25, 0.45, 0.65],
        },
        model_name="test",
    )
    report_path = save_text_report(results, model_name="test")

    print(f" Matrice de confusion  : {cm_path}")
    print(f" Courbes entrainement  : {hist_path}")
    print(f" Rapport texte         : {report_path}")
    print("\n reports.py operationnel")
