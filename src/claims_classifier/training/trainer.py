"""
Boucle d'entrainement pour les modeles de classification de texte.

Implements :
  - Passe forward + backward (retro-propagation, cours MLP)
  - Mise a jour des poids avec Adam 
  - Early stopping sur le Weighted F1 (validation)
  - Checkpointing du meilleur modele
  - Suivi TensorBoard (cours perceptron)
  - Gradient clipping (stabilite numerique, cours)
"""

import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from claims_classifier.config import config
from claims_classifier.utils import get_device

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Arrete l'entrainement si la metrique de validation ne s'ameliore plus.

    Surveille le Weighted F1 sur le jeu de validation.
    Si apres `patience` epoques consecutives il n'y a pas d'amelioration,
    l'entrainement est stoppe et le meilleur modele est restaure.

    """

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float = -float("inf")
        self.counter: int = 0
        self.best_state: dict | None = None

    def step(self, score: float, model: nn.Module) -> bool:
        """
        Met a jour l'etat de l'early stopping.

        Args:
            score : Weighted F1 sur la validation a l'epoque courante.
            model : Modele courant.

        Returns:
            True si l'entrainement doit s'arreter, False sinon.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            # Sauvegarder l'etat du meilleur modele en memoire
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Restaure les poids du meilleur modele."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info(f"Meilleur modele restaure (F1={self.best_score:.4f})")


class Trainer:
    """
    Gestionnaire d'entrainement pour MLP et TextCNN.

    Utilisation :
        trainer = Trainer(model, loss_fn, model_name="mlp")
        history = trainer.fit(train_loader, val_loader)

    Args:
        model      : Instance de MLP ou TextCNN.
        loss_fn    : CrossEntropyLoss ponderee (build_loss).
        model_name : Nom du modele pour les logs et sauvegardes ("mlp" ou "textcnn").
        device     : Dispositif de calcul (cpu ou cuda).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_classes: int,
        model_name: str = "model",
        device: torch.device | None = None,
    ) -> None:
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.model_name = model_name
        # Conserve pour passer labels=range(num_classes) a sklearn (regle anti-biais)
        self.num_classes = num_classes
        self._class_indices = list(range(num_classes))

        # Optimiseur Adam (pas adaptatif — cours optimisation numerique)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,  # regularisation L2 (cours)
        )

        # Chemins de sauvegarde
        self.checkpoint_path = (
            config.artifacts.models_dir / f"{model_name}_best.pt"
        )
        config.artifacts.models_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard (cours perceptron)
        run_dir = config.artifacts.runs_dir / model_name
        self.writer = SummaryWriter(log_dir=str(run_dir))
        logger.info(f"TensorBoard : {run_dir}")

    # -------------------------------------------------------------------------
    # Boucle d'entrainement
    # -------------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """
        Lance l'entrainement complet avec early stopping.

        Pour chaque epoque :
          1. Passe train  : forward -> loss -> backward -> optimizer.step
          2. Passe val    : forward uniquement (pas de gradient)
          3. Early stopping : si val_f1 ne s'ameliore plus -> stop
          4. TensorBoard  : log des metriques

        Args:
            train_loader : DataLoader du jeu d'entrainement.
            val_loader   : DataLoader du jeu de validation.

        Returns:
            Historique des metriques {metric_name: [valeurs par epoque]}.
        """
        early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience
        )

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
        }

        logger.info(
            f"\n{'='*60}\n"
            f"Entrainement : {self.model_name.upper()}\n"
            f"  Epoques max  : {config.training.num_epochs}\n"
            f"  Batch size   : {config.training.batch_size}\n"
            f"  LR           : {config.training.learning_rate}\n"
            f"  Weight decay : {config.training.weight_decay}\n"
            f"  Patience     : {config.training.early_stopping_patience}\n"
            f"  Dispositif   : {self.device}\n"
            f"{'='*60}"
        )

        for epoch in range(1, config.training.num_epochs + 1):
            t0 = time.time()

            # -- Passe TRAIN ---------------------------------------------------
            train_loss, train_f1 = self._train_epoch(train_loader)

            # -- Passe VALIDATION ----------------------------------------------
            val_loss, val_f1 = self._eval_epoch(val_loader)

            elapsed = time.time() - t0

            # -- Logging -------------------------------------------------------
            logger.info(
                f"Epoque {epoch:02d}/{config.training.num_epochs} | "
                f"{elapsed:.1f}s | "
                f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
            )

            # TensorBoard (cours)
            self.writer.add_scalars(
                "Loss", {"train": train_loss, "val": val_loss}, epoch
            )
            self.writer.add_scalars(
                "Weighted_F1", {"train": train_f1, "val": val_f1}, epoch
            )

            # Historique
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)

            # -- Early stopping ------------------------------------------------
            should_stop = early_stopping.step(val_f1, self.model)

            if should_stop:
                logger.info(
                    f"Early stopping declenche a l'epoque {epoch} "
                    f"(patience={config.training.early_stopping_patience})"
                )
                break

        # Restaurer et sauvegarder le meilleur modele
        early_stopping.restore_best(self.model)
        self._save_checkpoint(early_stopping.best_score)
        self.writer.close()

        return history

    # -------------------------------------------------------------------------
    # Passes individuelles
    # -------------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader
    ) -> tuple[float, float]:
        """
        Une epoque d'entrainement.

        Pour chaque batch :
          1. Forward : calcul des logits
          2. Loss    : comparaison logits / labels
          3. Backward: retro-propagation du gradient (cours)
          4. Clip    : ecreter les gradients (stabilite numerique, cours)
          5. Step    : mise a jour des poids (descente de gradient, cours)

        Returns:
            (loss_moyenne, weighted_f1) sur l'epoque.
        """
        self.model.train()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for input_ids, labels in loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Reinitialiser les gradients (sinon ils s'accumulent)
            self.optimizer.zero_grad()

            # Passe forward
            logits = self.model(input_ids)          # [batch, num_classes]
            loss = self.loss_fn(logits, labels)

            # Passe backward (retro-propagation — cours)
            loss.backward()

            # Gradient clipping (stabilite numerique — cours)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.training.gradient_clip_value
            )

            # Mise a jour des poids (Adam — cours optimisation)
            self.optimizer.step()

            total_loss += loss.item()

            # Predictions pour le calcul du F1
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(loader)
        weighted_f1 = f1_score(
            all_labels, all_preds,
            labels=self._class_indices,
            average="weighted", zero_division=0,
        )

        return avg_loss, weighted_f1

    def _eval_epoch(
        self, loader: DataLoader
    ) -> tuple[float, float]:
        """
        Une epoque d'evaluation (val ou test).

        Pas de backward ni de mise a jour des poids.
        torch.no_grad() desactive le calcul des gradients pour economiser
        la memoire et accelerer l'inference.

        Returns:
            (loss_moyenne, weighted_f1) sur l'epoque.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for input_ids, labels in loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input_ids)
                loss = self.loss_fn(logits, labels)

                total_loss += loss.item()

                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(loader)
        weighted_f1 = f1_score(
            all_labels, all_preds,
            labels=self._class_indices,
            average="weighted", zero_division=0,
        )

        return avg_loss, weighted_f1

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    def _save_checkpoint(self, best_f1: float) -> None:
        """
        Sauvegarde le modele et les metadonnees.

        On ne stocke pas l'optimiseur (pas de reprise d'entrainement prevue
        en Phase 1) — le checkpoint est destine a l'inference et reste leger.
        """
        checkpoint = {
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "num_classes": self.num_classes,
            "best_val_f1": best_f1,
            "config": config.model_dump(mode="json"),
        }
        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"Checkpoint sauvegarde : {self.checkpoint_path} (val_f1={best_f1:.4f})")

    def load_best(self) -> None:
        """Charge le meilleur checkpoint sauvegarde sur disque."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Checkpoint charge : {self.checkpoint_path} "
            f"(val_f1={checkpoint['best_val_f1']:.4f})"
        )