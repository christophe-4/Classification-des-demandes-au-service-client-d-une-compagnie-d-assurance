"""
Tests de fumee (smoke tests).

Verifient que :
  - La configuration se charge sans erreur
  - Les modules s'importent
  - Les modeles instancient et produisent des logits de la bonne forme
  - Le pipeline de nettoyage fonctionne sur des exemples synthetiques
  - Le vocabulaire encode/decode de facon coherente

Ces tests n'ont pas besoin du CSV reel : ils tournent en quelques secondes
et detectent les regressions critiques avant un entrainement complet.
"""

import pandas as pd
import torch

from claims_classifier.config import config
from claims_classifier.data.cleaning import clean_text, map_labels
from claims_classifier.data.dataset import LabelEncoder, make_splits
from claims_classifier.data.vocab import Vocabulary
from claims_classifier.models.mlp import MLP
from claims_classifier.models.textcnn import TextCNN
from claims_classifier.utils import get_device, set_seed

# =============================================================================
# CONFIG
# =============================================================================


def test_config_chargee():
    """La configuration globale est instanciee et coherente."""
    assert config.preprocessing.vocab_size == 30_000
    assert config.preprocessing.max_seq_length == 256
    assert config.split.train_ratio + config.split.val_ratio + config.split.test_ratio == 1.0
    assert config.training.seed == 42


# =============================================================================
# UTILS
# =============================================================================


def test_set_seed_reproductible():
    """set_seed produit des tirages identiques."""
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_get_device_retourne_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


# =============================================================================
# CLEANING
# =============================================================================


def test_clean_text_minuscules_et_tokens():
    """Le nettoyage applique minuscules, dates et montants."""
    raw = "I paid $1,200.50 on XX/XX/2024 with my XXXX card.\nThanks!"
    cleaned = clean_text(raw)
    assert cleaned == cleaned.lower()
    assert "<money>" in cleaned
    assert "<date>" in cleaned
    assert "xxxx" not in cleaned
    assert "\n" not in cleaned


def test_map_labels_fusion_21_vers_12():
    """Les libelles bruts sont bien fusionnes."""
    df = pd.DataFrame(
        {
            "label": [
                "Credit reporting",
                "Credit reporting, credit repair services, or other personal consumer reports",
                "Credit card",
                "Mortgage",
            ],
            "text": ["a", "b", "c", "d"],
        }
    )
    out = map_labels(df)
    assert set(out["label"]) == {"credit_reporting", "credit_card", "mortgage"}


# =============================================================================
# VOCAB
# =============================================================================


def test_vocab_encode_decode_aller_retour():
    """Encode puis decode retourne les memes mots connus."""
    texts = pd.Series(
        [
            "credit report violation federal law",
            "credit report account",
            "federal law account",
        ]
    )
    vocab = Vocabulary.build(texts, vocab_size=100, min_frequency=1)
    ids = vocab.encode("credit report account")
    decoded = vocab.decode(ids)
    assert decoded == ["credit", "report", "account"]


def test_vocab_unknown_to_unk():
    """Les mots inconnus sont mappes sur <unk>."""
    texts = pd.Series(["credit report"])
    vocab = Vocabulary.build(texts, vocab_size=10, min_frequency=1)
    ids = vocab.encode("totalementinconnu")
    assert ids == [vocab.unk_id]


# =============================================================================
# SPLITS
# =============================================================================


def test_make_splits_proportions():
    """Les proportions train/val/test correspondent a la config."""
    df = pd.DataFrame(
        {
            "text": [f"text {i}" for i in range(1000)],
            # 4 classes pour permettre la stratification
            "label": (["a"] * 250 + ["b"] * 250 + ["c"] * 250 + ["d"] * 250),
        }
    )
    train, val, test = make_splits(df)
    n = len(df)
    assert abs(len(train) / n - config.split.train_ratio) < 0.02
    assert abs(len(val) / n - config.split.val_ratio) < 0.02
    assert abs(len(test) / n - config.split.test_ratio) < 0.02
    # Aucun chevauchement (verifie via les textes uniques)
    assert set(train["text"]).isdisjoint(set(val["text"]))
    assert set(train["text"]).isdisjoint(set(test["text"]))
    assert set(val["text"]).isdisjoint(set(test["text"]))


# =============================================================================
# LABEL ENCODER
# =============================================================================


def test_label_encoder_ordre_alphabetique():
    """L'encodeur trie les classes par ordre alphabetique (reproductibilite)."""
    labels = pd.Series(["mortgage", "credit_card", "debt_collection"])
    enc = LabelEncoder.build(labels)
    assert enc.classes == ["credit_card", "debt_collection", "mortgage"]
    assert enc.encode("credit_card") == 0
    assert enc.decode(0) == "credit_card"


# =============================================================================
# MODELES
# =============================================================================


def test_mlp_forward_shape():
    """MLP retourne des logits [batch, num_classes]."""
    vocab_size = 100
    num_classes = 12
    model = MLP(vocab_size=vocab_size, num_classes=num_classes)
    fake_input = torch.randint(0, vocab_size, (4, 32))
    logits = model(fake_input)
    assert logits.shape == (4, num_classes)
    assert logits.dtype == torch.float32


def test_textcnn_forward_shape():
    """TextCNN retourne des logits [batch, num_classes]."""
    vocab_size = 100
    num_classes = 12
    model = TextCNN(vocab_size=vocab_size, num_classes=num_classes)
    # seq_len >= max(kernel_sizes) sinon la convolution echoue
    fake_input = torch.randint(0, vocab_size, (4, 32))
    logits = model(fake_input)
    assert logits.shape == (4, num_classes)


def test_mlp_dropout_zero_accepte():
    """dropout=0.0 doit etre accepte (non remplace par la config)."""
    model = MLP(vocab_size=50, num_classes=3, dropout=0.0)
    # Verifier qu'aucune couche Dropout n'a un p > 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.p == 0.0
