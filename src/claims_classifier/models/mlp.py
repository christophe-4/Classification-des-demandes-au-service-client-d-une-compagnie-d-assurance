# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Modele MLP pour la classification de texte.

Architecture :
  1. Embedding     : token IDs -> vecteurs denses (appris from scratch)
  2. AvgPool       : moyenne des embeddings -> vecteur fixe (independant de la longueur)
  3. Dense + ReLU  : couche cachee (cours : fonction d'activation)
  4. Dropout       : regularisation (cours : weight decay / regularisation)
  5. Dense         : couche de sortie -> logits pour chaque classe

"""

import torch
import torch.nn as nn

from claims_classifier.config import config


class MLP(nn.Module):
    """
    MLP sur embeddings moyens pour la classification de texte.

    Args:
        vocab_size  : Taille du vocabulaire (config.preprocessing.vocab_size + 2).
        num_classes : Nombre de classes (12).
        embed_dim   : Dimension des embeddings (config.model.embedding_dim).
        hidden_dim  : Dimension de la couche cachee (config.model.mlp_hidden_dim).
        dropout     : Taux de dropout (config.model.mlp_dropout).
        pad_idx     : ID du token <pad> — ses embeddings restent a zero.

    Exemple :
        model = MLP(vocab_size=30002, num_classes=12)
        logits = model(input_ids)  # [batch, 12]
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout: float | None = None,
        pad_idx: int | None = None,
    ) -> None:
        super().__init__()

        embed_dim = embed_dim if embed_dim is not None else config.model.embedding_dim
        hidden_dim = hidden_dim if hidden_dim is not None else config.model.mlp_hidden_dim
        dropout = dropout if dropout is not None else config.model.mlp_dropout
        pad_idx = pad_idx if pad_idx is not None else config.preprocessing.pad_token_id

        # -- Couche 1 : Embedding ----------------------------------------------
        # Transforme chaque token ID en vecteur dense de dimension embed_dim.
        # padding_idx=pad_idx : le vecteur du token <pad> reste toujours a zero
        # et ne contribue pas au gradient (bon comportement pour le padding).
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # -- Couches 2-4 : MLP classique (cours perceptron multi-couches) ------
        self.classifier = nn.Sequential(
            # Dense 1 : embed_dim -> hidden_dim
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),  # activation non-lineaire (cours)
            nn.Dropout(p=dropout),  # regularisation (cours)
            # Dense 2 : hidden_dim -> num_classes (logits)
            nn.Linear(hidden_dim, num_classes),
            # Pas de Softmax ici : nn.CrossEntropyLoss l'applique en interne
            # (plus stable numeriquement — cours stabilite numerique)
        )

        # Initialisation de Xavier (cours stabilite numerique)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialisation de Xavier pour les couches lineaires (cours)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(input_ids)

        # Moyenne masquee : on exclut les tokens <pad> (ID=0)
        # Masque : True sur les vrais tokens, False sur le padding
        # [batch, seq_len] -> [batch, seq_len, 1]
        mask = (input_ids != self.embedding.padding_idx).float().unsqueeze(-1)

        # Somme des embeddings reels / nombre de vrais tokens
        # Evite la division par zero (texte entierement vide -> impossible
        # apres nettoyage, mais on securise avec clamp)
        sum_embedded = (embedded * mask).sum(dim=1)  # [batch, embed_dim]
        lengths = mask.sum(dim=1).clamp(min=1.0)  # [batch, 1]
        pooled = sum_embedded / lengths  # [batch, embed_dim]

        # [batch, embed_dim] -> [batch, num_classes]
        logits = self.classifier(pooled)

        return logits

    def count_parameters(self) -> int:
        """Nombre de parametres entrainables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Simuler vocab et encoder
    vocab_size = config.preprocessing.vocab_size + 2  # +2 pour <pad> et <unk>
    num_classes = 12

    model = MLP(vocab_size=vocab_size, num_classes=num_classes)

    print(model)
    print(f"\nParametres entrainables : {model.count_parameters():,}")

    # Verifier la passe forward avec un batch fictif
    batch_size = 4
    seq_len = config.preprocessing.max_seq_length
    fake_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(fake_input)

    print("\nVerification passe forward :")
    print(f"  Input  : {fake_input.shape}")  # [4, 256]
    print(f"  Output : {logits.shape}")  # [4, 12]
    print(f"  dtype  : {logits.dtype}")  # float32
    print("\n MLP operationnel")
