"""
Gradio demo — Claims Classifier (TextCNN from scratch, Weighted F1 = 83.12 %)
v3 — self-contained, aucune dépendance sur le package claims-classifier

Utilisation locale :
    $env:LOCAL_MODE="true"
    uv run python hf_space/app.py
"""

import json
import logging
import os
import re
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
HF_MODEL_REPO  = "FrenchEdtech/claims-classifier"
DEVICE         = torch.device("cpu")
MAX_SEQ_LENGTH = 256   # config.preprocessing.max_seq_length
PAD_ID         = 0     # config.preprocessing.pad_token_id
UNK_ID         = 1     # config.preprocessing.unk_token_id
TOP_K          = 5

LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"


# ── Nettoyage du texte (porté depuis data/cleaning.py) ────────────────────────
_RE_DATE      = re.compile(r"\b(?:xx|x){1,2}/(?:xx|x){1,2}/(?:xx|x|\d){2,4}\b|\bxx/xx/year\b")
_RE_MONEY     = re.compile(r"\{?\$\s?[\d,]+\.?\d*\}?")
_RE_XXXX      = re.compile(r"\bx{2,}\b")
_RE_NEWLINE   = re.compile(r"\\n|\n|\r")
_RE_NON_ALPHA = re.compile(r"[^a-z\s<>]")
_RE_SPACE     = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Nettoyage identique à data/cleaning.py : minuscules, dates, montants, XXXX."""
    text = text.lower()
    text = _RE_NEWLINE.sub(" ", text)
    text = _RE_DATE.sub(" <date> ", text)
    text = _RE_MONEY.sub(" <money> ", text)
    text = _RE_XXXX.sub(" ", text)
    text = _RE_NON_ALPHA.sub(" ", text)
    return _RE_SPACE.sub(" ", text).strip()


# ── Vocabulaire (porté depuis data/vocab.py) ──────────────────────────────────
class Vocabulary:
    """Chargement et encodage du vocabulaire depuis vocab.json."""

    def __init__(self, word2idx: dict[str, int]) -> None:
        self.word2idx = word2idx

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        logger.info(f"Vocabulaire chargé : {len(word2idx):,} mots")
        return cls(word2idx)

    def encode(self, text: str) -> list[int]:
        return [self.word2idx.get(w, UNK_ID) for w in text.split()]

    def __len__(self) -> int:
        return len(self.word2idx)


# ── Label encoder (porté depuis data/dataset.py) ──────────────────────────────
class LabelEncoder:
    """Chargement et décodage des labels depuis label_encoder.json."""

    def __init__(self, label2idx: dict[str, int]) -> None:
        self.label2idx = label2idx
        self.idx2label = {idx: label for label, idx in label2idx.items()}

    @classmethod
    def load(cls, path: Path) -> "LabelEncoder":
        with open(path, "r", encoding="utf-8") as f:
            label2idx = json.load(f)
        logger.info(f"LabelEncoder chargé : {len(label2idx)} classes")
        return cls(label2idx)

    def decode(self, idx: int) -> str:
        return self.idx2label[idx]

    @property
    def num_classes(self) -> int:
        return len(self.label2idx)


# ── TextCNN (porté depuis models/textcnn.py) ──────────────────────────────────
class TextCNN(nn.Module):
    """
    CNN 1D pour la classification de texte (Kim 2014).
    Architecture identique à models/textcnn.py — hyperparamètres par défaut.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).permute(0, 2, 1)
        pooled   = [F.relu(conv(embedded)).max(dim=2).values for conv in self.convolutions]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


# ── Résolution des chemins d'artefacts ────────────────────────────────────────
if LOCAL_MODE:
    _root       = Path(__file__).resolve().parent.parent
    _ckpt_path  = _root / "models"  / "textcnn_best.pt"
    _vocab_path = _root / "data"    / "processed" / "vocab.json"
    _le_path    = _root / "data"    / "processed" / "label_encoder.json"
    logger.info("LOCAL_MODE — chargement depuis les fichiers locaux")
else:
    logger.info("HF Hub — téléchargement des artefacts (mis en cache)...")
    _ckpt_path  = Path(hf_hub_download(HF_MODEL_REPO, "textcnn_best.pt"))
    _vocab_path = Path(hf_hub_download(HF_MODEL_REPO, "vocab.json"))
    _le_path    = Path(hf_hub_download(HF_MODEL_REPO, "label_encoder.json"))


# ── Chargement au démarrage — une seule fois ──────────────────────────────────
vocab         = Vocabulary.load(_vocab_path)
label_encoder = LabelEncoder.load(_le_path)
ckpt          = torch.load(_ckpt_path, map_location=DEVICE, weights_only=True)
model         = TextCNN(vocab_size=len(vocab), num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
logger.info(f"Modèle prêt — {ckpt['num_classes']} classes · val Weighted F1 = {ckpt['best_val_f1']:.4f}")


# ── Fonction de prédiction ────────────────────────────────────────────────────
def predict(text: str) -> dict[str, float]:
    """Classifie une réclamation client en 12 catégories financières."""
    if not text or len(text.strip()) < 3:
        return {"(Veuillez saisir un texte)": 1.0}

    cleaned = clean_text(text)
    if not cleaned.strip():
        return {"(Texte vide après nettoyage)": 1.0}

    ids = vocab.encode(cleaned)[:MAX_SEQ_LENGTH]
    ids += [PAD_ID] * (MAX_SEQ_LENGTH - len(ids))

    with torch.no_grad():
        probs = torch.softmax(
            model(torch.tensor([ids], dtype=torch.long, device=DEVICE)),
            dim=1
        )[0]

    return {label_encoder.decode(i): float(probs[i]) for i in range(label_encoder.num_classes)}


# ── Interface Gradio ──────────────────────────────────────────────────────────
EXAMPLES = [
    ["I have an error on my credit report that is not mine. "
     "The account shows a balance I do not owe and I never opened this account."],
    ["My mortgage servicer incorrectly applied my monthly payment "
     "and is now charging late fees even though I paid on time."],
    ["A debt collection agency keeps calling me five times a day "
     "about a debt that I already paid off six months ago."],
]

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
        "📂 [Code source GitHub]"
        "(https://github.com/christophe-4/Classification-des-demandes-au-service-client-d-une-compagnie-d-assurance)"
    ),
    examples=EXAMPLES,
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
