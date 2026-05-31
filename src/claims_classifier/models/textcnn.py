# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Modele TextCNN pour la classification de texte.

Architecture (adaptation 1D du cours CNN) :
  1. Embedding      : token IDs -> vecteurs denses (appris from scratch)
  2. Conv1D x3      : filtres de tailles 3, 4, 5 en parallele
                      (equivalent des filtres 2D du cours, appliques sur texte)
  3. GlobalMaxPool  : valeur maximale par filtre (cours : max pooling)
  4. Concat         : concatenation des 3 sorties -> vecteur unifie
  5. Dropout        : regularisation (cours)
  6. Dense          : logits pour chaque classe

Pourquoi plusieurs tailles de kernels ?
  kernel=3 detecte les trigrammes  (ex: "credit report violation")
  kernel=4 detecte les quadrigrammes (ex: "federal law credit bureau")
  kernel=5 detecte les pentagrammes (ex: "dispute account credit report agency")
  En concatenant les 3, le modele capture des motifs de longueurs variees,
  comme un CNN qui detecte des textures a differentes echelles (cours).

Reference : Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from claims_classifier.config import config


class TextCNN(nn.Module):
    """
    CNN 1D pour la classification de texte (Kim 2014).

    Args:
        vocab_size   : Taille du vocabulaire.
        num_classes  : Nombre de classes (12).
        embed_dim    : Dimension des embeddings.
        num_filters  : Nombre de filtres par taille de kernel.
        kernel_sizes : Tailles des kernels (defaut: (3, 4, 5)).
        dropout      : Taux de dropout.
        pad_idx      : ID du token <pad>.

    Exemple :
        model = TextCNN(vocab_size=30002, num_classes=12)
        logits = model(input_ids)  # [batch, 12]
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int | None = None,
        num_filters: int | None = None,
        kernel_sizes: tuple[int, ...] | None = None,
        dropout: float | None = None,
        pad_idx: int | None = None,
    ) -> None:
        super().__init__()

        embed_dim = embed_dim if embed_dim is not None else config.model.embedding_dim
        num_filters = num_filters if num_filters is not None else config.model.cnn_num_filters
        kernel_sizes = kernel_sizes if kernel_sizes is not None else config.model.cnn_kernel_sizes
        dropout = dropout if dropout is not None else config.model.cnn_dropout
        pad_idx = pad_idx if pad_idx is not None else config.preprocessing.pad_token_id

        self.kernel_sizes = kernel_sizes

        # -- Couche 1 : Embedding ----------------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # -- Couches 2 : Convolutions 1D en parallele -------------------------
        # Une couche Conv1D par taille de kernel.
        # Chaque Conv1D detecte des motifs de longueur kernel_size dans la sequence.
        #
        # Dimensions :
        #   Input  : [batch, seq_len, embed_dim]
        #   -> apres permute : [batch, embed_dim, seq_len]  (format Conv1D)
        #   Conv1D(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
        #   -> [batch, num_filters, seq_len - k + 1]
        #   GlobalMaxPool
        #   -> [batch, num_filters]
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=0,  # pas de padding : on veut reduire la sequence
                )
                for k in kernel_sizes
            ]
        )

        # -- Couches 3-4 : Dropout + Dense ------------------------------------
        # Taille d'entree = num_filters * nombre de kernels
        # (on concatene les sorties de chaque Conv1D)
        total_filters = num_filters * len(kernel_sizes)

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(total_filters, num_classes)

        # Initialisation de Xavier (cours stabilite numerique)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialisation de Xavier pour les couches lineaires et convolutives."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Passe forward du TextCNN.

        Args:
            input_ids: Tenseur de token IDs [batch_size, seq_len].

        Returns:
            logits: Tenseur de scores bruts [batch_size, num_classes].
        """
        # -- Embedding ---------------------------------------------------------
        # [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(input_ids)

        # Conv1D attend [batch, channels, seq_len]
        # -> [batch, embed_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        # -- Convolutions + GlobalMaxPool --------------------------------------
        # Pour chaque taille de kernel :
        #   1. Convolution 1D -> [batch, num_filters, seq_len - k + 1]
        #   2. ReLU (activation non-lineaire, cours)
        #   3. GlobalMaxPool -> [batch, num_filters]
        #      (valeur max sur toute la sequence = motif le plus actif)
        pooled_outputs = []
        for conv in self.convolutions:
            # [batch, num_filters, seq_len - k + 1]
            conv_out = F.relu(conv(embedded))

            # GlobalMaxPool : max sur la dimension temporelle
            # [batch, num_filters, L] -> [batch, num_filters]
            pooled = conv_out.max(dim=2).values
            pooled_outputs.append(pooled)

        # -- Concatenation -----------------------------------------------------
        # [batch, num_filters] x 3 -> [batch, num_filters * 3]
        concatenated = torch.cat(pooled_outputs, dim=1)

        # -- Dropout + Classification ------------------------------------------
        dropped = self.dropout(concatenated)
        logits = self.classifier(dropped)

        return logits

    def count_parameters(self) -> int:
        """Nombre de parametres entrainables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    vocab_size = config.preprocessing.vocab_size + 2
    num_classes = 12

    model = TextCNN(vocab_size=vocab_size, num_classes=num_classes)

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

    # Verifier les dimensions intermediaires
    print("\nDimensions intermediaires :")
    embedded = model.embedding(fake_input).permute(0, 2, 1)
    print(f"  Embedding (apres permute) : {embedded.shape}")  # [4, 128, 256]
    for i, (conv, k) in enumerate(zip(model.convolutions, model.kernel_sizes)):
        out = torch.relu(conv(embedded))
        pooled = out.max(dim=2).values
        print(f"  Conv1D(kernel={k}) -> {out.shape} -> MaxPool -> {pooled.shape}")

    print("\n TextCNN operationnel")
