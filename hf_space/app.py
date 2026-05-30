"""
Gradio demo — Claims Classifier (TextCNN from scratch, Weighted F1 = 83.12 %)
v2 — fix Python >=3.11 constraint

Utilisation :
    # Test local (artefacts dans models/ et data/processed/)
    $env:LOCAL_MODE="true"
    uv run python hf_space/app.py

    # Production HF Space (artefacts sur HF Hub)
    python app.py
"""

import logging
import os
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from claims_classifier.data.cleaning import clean_text
from claims_classifier.data.dataset import LabelEncoder
from claims_classifier.data.vocab import Vocabulary
from claims_classifier.models.textcnn import TextCNN

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
HF_MODEL_REPO  = "FrenchEdtech/claims-classifier"
DEVICE         = torch.device("cpu")
MAX_SEQ_LENGTH = 256   # config.preprocessing.max_seq_length
PAD_ID         = 0     # config.preprocessing.pad_token_id
TOP_K          = 5

LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

# ── Résolution des chemins d'artefacts ────────────────────────────────────────
if LOCAL_MODE:
    _root       = Path(__file__).resolve().parent.parent
    _ckpt_path  = _root / "models"  / "textcnn_best.pt"
    _vocab_path = _root / "data"    / "processed" / "vocab.json"
    _le_path    = _root / "data"    / "processed" / "label_encoder.json"
    logger.info("LOCAL_MODE — chargement depuis les fichiers locaux")
else:
    logger.info("HF Hub — téléchargement des artefacts (mise en cache)...")
    _ckpt_path  = Path(hf_hub_download(HF_MODEL_REPO, "textcnn_best.pt"))
    _vocab_path = Path(hf_hub_download(HF_MODEL_REPO, "vocab.json"))
    _le_path    = Path(hf_hub_download(HF_MODEL_REPO, "label_encoder.json"))

# ── Chargement au démarrage — une seule fois (pas de rechargement par requête) ─
logger.info("Chargement du vocabulaire et du label encoder...")
vocab         = Vocabulary.load(_vocab_path)
label_encoder = LabelEncoder.load(_le_path)

logger.info("Reconstruction et chargement du checkpoint TextCNN...")
ckpt  = torch.load(_ckpt_path, map_location=DEVICE, weights_only=True)
model = TextCNN(vocab_size=len(vocab), num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
logger.info(
    f"Modèle prêt — {ckpt['num_classes']} classes · "
    f"val Weighted F1 = {ckpt['best_val_f1']:.4f}"
)


# ── Fonction de prédiction ────────────────────────────────────────────────────
def predict(text: str) -> dict[str, float]:
    """Classifie une réclamation client en 12 catégories financières."""
    if not text or len(text.strip()) < 3:
        return {"(Veuillez saisir un texte)": 1.0}

    cleaned = clean_text(text)
    if not cleaned or not cleaned.strip():
        return {"(Texte vide après nettoyage)": 1.0}

    # Encodage + troncature + padding
    ids = vocab.encode(cleaned)[:MAX_SEQ_LENGTH]
    ids = ids + [PAD_ID] * (MAX_SEQ_LENGTH - len(ids))

    input_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids)
        probs  = torch.softmax(logits, dim=1)[0]

    # Retourne un dict label -> probabilité (Gradio affiche top-k automatiquement)
    return {
        label_encoder.decode(i): float(probs[i])
        for i in range(label_encoder.num_classes)
    }


# ── Exemples préremplis ───────────────────────────────────────────────────────
EXAMPLES = [
    [
        "I have an error on my credit report that is not mine. "
        "The account shows a balance I do not owe and I never opened this account. "
        "I have contacted the bureau multiple times but the error remains."
    ],
    [
        "My mortgage servicer incorrectly applied my monthly payment "
        "and is now charging late fees even though I paid on time. "
        "This is affecting my credit score."
    ],
    [
        "A debt collection agency keeps calling me five times a day "
        "about a debt that I already paid off six months ago. "
        "I have proof of payment but they continue to harass me."
    ],
]


# ── Interface Gradio ──────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Réclamation client (en anglais)",
        placeholder="Describe your financial complaint here...",
        lines=6,
    ),
    outputs=gr.Label(
        label="Classification (top 5 catégories)",
        num_top_classes=TOP_K,
    ),
    title="🏦 Claims Classifier — Classification de réclamations clients",
    description=(
        "**TextCNN** *from scratch* · Weighted F1 = **83.12 %** · 12 classes financières\n\n"
        "Entraîné sur 300 000 réclamations CFPB (Consumer Financial Protection Bureau). "
        "Le modèle classe automatiquement une réclamation vers le bon département.\n\n"
        "📂 [Code source GitHub](https://github.com/christophe-4/Classification-des-demandes-au-service-client-d-une-compagnie-d-assurance)"
    ),
    examples=EXAMPLES,
    cache_examples=False,
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
