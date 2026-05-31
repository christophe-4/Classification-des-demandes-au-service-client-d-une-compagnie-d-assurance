# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Schemas Pydantic pour les requetes et reponses de l'API.

Ces modeles assurent la validation automatique des entrees/sorties
et alimentent la documentation interactive (/docs).
"""

from pydantic import BaseModel, Field

# =============================================================================
# REQUETES
# =============================================================================


class PredictRequest(BaseModel):
    """Corps d'une requete de prediction."""

    text: str = Field(
        ...,
        min_length=10,
        description="Reclamation client a classifier (en anglais, min 10 caracteres).",
        examples=[
            "I have been trying to remove an incorrect account "
            "from my credit report for months without success"
        ],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Nombre de predictions a retourner (entre 1 et 12).",
    )


# =============================================================================
# REPONSES
# =============================================================================


class PredictionItem(BaseModel):
    """Une prediction unitaire : classe + probabilite."""

    class_name: str = Field(..., description="Nom de la categorie predite.")
    probability: float = Field(..., description="Probabilite associee (arrondie a 4 decimales).")


class PredictResponse(BaseModel):
    """Reponse complete d'une prediction."""

    prediction: str = Field(..., description="Classe la plus probable.")
    confidence: float = Field(..., description="Probabilite de la classe principale (4 decimales).")
    top_k: list[PredictionItem] = Field(
        ..., description="Liste des k meilleures predictions, triees par probabilite."
    )
    model_name: str = Field(..., description="Architecture utilisee (ex: textcnn).")
    weighted_f1: float = Field(
        ..., description="Weighted F1 de validation du checkpoint (ex: 0.8276)."
    )
    inference_time_ms: float = Field(..., description="Temps d'inference en millisecondes.")


class HealthResponse(BaseModel):
    """Reponse du endpoint de sante."""

    status: str = Field(..., description="Statut de l'API ('ok').")
    model_loaded: bool = Field(..., description="True si le modele est charge en memoire.")
    model_name: str = Field(..., description="Architecture chargee (ex: textcnn).")
    num_classes: int = Field(..., description="Nombre de classes du modele.")
    weighted_f1: float = Field(..., description="Weighted F1 de validation du checkpoint.")
