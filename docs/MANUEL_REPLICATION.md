# Manuel de réplication — Classifieur de texte pour PME

> **Usage** : ce manuel accompagne le template
> [`claims-classifier`](https://github.com/christophe-4/Classification-des-demandes-au-service-client-d-une-compagnie-d-assurance).
> Il décrit, étape par étape, comment l'adapter à un nouveau dataset client.
> Il ne contient pas de code — uniquement des pointeurs vers les fichiers à modifier.

---

## 0. À qui s'adresse ce manuel

Tu es consultant (Supply Chain, Ops, ou IA appliquée) et tu veux déployer
un classificateur de texte pour une PME cliente — ticketing, réclamations,
e-mails entrants, devis, incidents terrain.

Ce template a été construit pour être réutilisé :
- **Structure modulaire** : chaque responsabilité vit dans son propre fichier
- **Zéro dépendance externe** : JSONL, SQLite absent, pas de cloud obligatoire
- **Un seul paramètre d'entrée** : un CSV avec une colonne texte + une colonne label

**Ce que ce manuel ne fait pas** : expliquer le Deep Learning.
Pour ça, lire les docstrings dans `src/claims_classifier/`.

---

## 1. Prérequis & installation

### Outils nécessaires

| Outil | Pourquoi | Installation |
|-------|----------|-------------|
| **Python 3.11** | Runtime du projet | [python.org](https://python.org) |
| **uv** | Gestionnaire de paquets ultra-rapide | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Git** | Clonage et versioning | [git-scm.com](https://git-scm.com) |
| **GPU NVIDIA** | Entraînement (optionnel, ~10× plus rapide) | Drivers CUDA 12.1 |
| **Docker** | Déploiement conteneurisé (optionnel) | [docker.com](https://docker.com) |

### Partir d'une base propre

```bash
# Cloner le template
git clone https://github.com/christophe-4/Classification-des-demandes-au-service-client-d-une-compagnie-d-assurance.git
cd Classification-des-demandes-au-service-client-d-une-compagnie-d-assurance

# Créer l'environnement et installer les dépendances
uv sync

# Vérifier
uv run python -c "import torch; print('OK —', torch.__version__)"
```

> **Astuce** : renomme immédiatement le dossier cloné avec le nom du client
> pour éviter toute confusion entre projets.

**Référence** : section *Installation rapide* dans `README.md`.

---

## 2. Comprendre l'architecture en 5 minutes

### Flux de données

```
CSV client
  │
  ├─ loader.py       → charge le CSV, renomme les colonnes
  ├─ cleaning.py     → nettoie le texte, mappe les labels
  ├─ vocab.py        → construit le vocabulaire (sur le TRAIN uniquement)
  ├─ dataset.py      → crée les splits 70/15/15 et les DataLoaders
  │
  ├─ models/         → entraîne MLP ou TextCNN
  ├─ evaluation/     → Weighted F1, matrice de confusion
  │
  ├─ api/            → expose POST /predict et GET /health
  ├─ Dockerfile      → conteneurise l'API
  └─ monitoring/     → logs JSONL, dérive, dashboard Streamlit
```

### Tableau de navigation rapide

| Je veux modifier… | Fichier à ouvrir | Ce qu'on change |
|-------------------|-----------------|-----------------|
| Noms de colonnes du CSV | `src/claims_classifier/data/loader.py` | `COL_TEXT`, `COL_LABEL` |
| Fusion/mapping des classes | `src/claims_classifier/data/cleaning.py` | `LABEL_MAPPING` (dict) |
| Regex de nettoyage | `src/claims_classifier/data/cleaning.py` | `clean_text()` |
| Taille du vocabulaire | `src/claims_classifier/config.py` | `vocab_size` |
| Longueur max des séquences | `src/claims_classifier/config.py` | `max_seq_length` |
| Architecture du modèle | `src/claims_classifier/config.py` | section `ModelConfig` |
| Hyperparamètres d'entraînement | `src/claims_classifier/config.py` | section `TrainingConfig` |
| Métriques d'évaluation | `src/claims_classifier/evaluation/metrics.py` | `evaluate()` |
| Endpoint de prédiction | `api/routers/predict.py` | fonction `predict()` |
| Seuils de dérive | `src/claims_classifier/config.py` | section `MonitoringConfig` |

---

## 3. Adapter au nouveau dataset (ÉTAPE CRITIQUE)

C'est l'étape où 80 % du travail d'adaptation se passe.
Elle ne demande que des modifications dans 1 à 3 fichiers.

### 3.1 Format attendu du CSV

Le projet attend un CSV avec **au moins deux colonnes** :
- Une colonne de **texte libre** (la réclamation, l'e-mail, le ticket…)
- Une colonne de **label** (la catégorie à prédire)

Les colonnes peuvent s'appeler n'importe comment — c'est ce qu'on adapte
à l'étape suivante.

**Où c'est défini** :
`src/claims_classifier/data/loader.py` — constantes `COL_TEXT` et `COL_LABEL` en haut du fichier.

**Contrainte** : le texte doit être en clair (pas d'encodage base64, pas de HTML brut).
Si le CSV contient du HTML, ajouter une étape de strip HTML dans `clean_text()`.

### 3.2 Adapter les noms de colonnes

Ouvrir `src/claims_classifier/data/loader.py` et changer les deux constantes
au sommet du fichier :

- `COL_TEXT` : nom exact de la colonne texte dans le CSV client
- `COL_LABEL` : nom exact de la colonne label dans le CSV client

C'est **tout** — le reste de la chaîne utilise ensuite `"text"` et `"label"`
après renommage automatique.

### 3.3 Adapter le mapping/fusion des classes

**Fichier** : `src/claims_classifier/data/cleaning.py` — dictionnaire `LABEL_MAPPING`.

`LABEL_MAPPING` est un dict `{label_brut_dans_CSV: label_final}`.
Il sert à :
1. **Fusionner** des classes redondantes (ex: trois libellés pour la même catégorie)
2. **Normaliser** les noms (espaces, casse, caractères spéciaux)
3. **Regrouper** les classes ultra-rares dans une catégorie `"other"` ou les supprimer

**Comment le réécrire** :

1. Lancer d'abord `df["label"].value_counts()` dans le notebook EDA
   pour voir tous les libellés bruts
2. Lister les classes finales souhaitées (idéalement 5–20 classes)
3. Écrire le dict : chaque libellé brut → classe finale
4. Les libellés absents du mapping sont **supprimés** (avec warning dans les logs)

**Règle empirique** : si une classe a moins de 200 observations dans le dataset
complet, la regrouper ou la supprimer — elle ne sera pas apprise correctement.

### 3.4 Adapter le nettoyage du texte

**Fichier** : `src/claims_classifier/data/cleaning.py` — fonction `clean_text()`.

Le nettoyage actuel est conçu pour des réclamations financières en anglais.
Voici ce que chaque regex fait et quand la garder ou l'adapter :

| Regex | Ce qu'elle fait | Garder si… | Adapter si… |
|-------|----------------|-----------|-------------|
| `RE_DATE` | Dates `XX/XX/XXXX` → `<date>` | Le dataset contient des dates anonymisées CFPB | Adapter le pattern si les dates sont au format JJ/MM/AAAA |
| `RE_MONEY` | Montants `$1,200` → `<money>` | Données financières | Changer le symbole pour `€`, `CHF`, etc. |
| `RE_XXXX` | Séquences `XXXX` → espace | Anonymisation CFPB | Supprimer si le dataset n'est pas anonymisé ainsi |
| `RE_NON_ALPHA` | Conserve lettres, espaces, `<` `>` | Toujours utile | Adapter si le domaine utilise des codes (`P-123`) |
| `RE_MULTISPACE` | Espaces multiples → un seul | Toujours | — |

**Cas particuliers fréquents en PME** :
- **Dataset médical** : conserver les codes CIM-10, supprimer les montants
- **Dataset juridique** : conserver les numéros d'article, adapter les dates
- **E-mails** : ajouter une étape `strip_email_header()` avant `clean_text()`
- **Devis/factures** : conserver les références produits

### 3.5 Passer au français (ou autre langue)

Ce template a été entraîné sur de l'anglais.
Pour du **français**, les changements sont mineurs :

1. **`clean_text()`** dans `cleaning.py` : les regex sont language-agnostiques
   (elles travaillent sur des patterns, pas sur des mots). Aucun changement
   nécessaire si les montants et dates ont le même format.

2. **`vocab.py`** — `Vocabulary.build()` : la tokenisation est par espace.
   Fonctionne pour le français sans modification, mais les contractions
   (`l'`, `d'`, `qu'`) ne seront pas splitées — c'est acceptable pour un
   TextCNN sur des tickets courts.

3. **`config.py`** — `vocab_size` : le français a plus de formes fléchies
   que l'anglais (conjugaisons). Augmenter légèrement à 35 000–40 000.

4. **`max_seq_length`** : les textes français sont souvent légèrement plus longs.
   Ajuster selon les percentiles observés dans l'EDA.

> **Pour des langues asiatiques** (chinois, japonais) : la tokenisation par
> espace ne fonctionne pas. Il faudra remplacer `text.split()` dans `vocab.py`
> et `clean_text()` par un tokeniseur adapté (jieba, MeCab…).

---

## 4. Analyse exploratoire

**Notebook** : `notebooks/01_eda.ipynb`

Lancer avec :
```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

### Ce qu'il faut vérifier AVANT d'entraîner

#### 4.1 Distribution des classes

| Observation | Interprétation | Action |
|-------------|----------------|--------|
| Une classe > 50 % | Déséquilibre fort | La loss pondérée est déjà configurée — vérifier les poids dans `losses.py` |
| Ratio max/min > 100:1 | Déséquilibre extrême | Fusionner ou supprimer les classes ultra-minoritaires |
| Une classe < 100 obs. | Trop rare | Regrouper dans `"other"` via `LABEL_MAPPING` |
| Classes équilibrées | Situation idéale | Aucune action particulière |

#### 4.2 Longueurs de texte

Regarder les percentiles p50, p75, p95.

| Observation | Action recommandée |
|-------------|-------------------|
| p95 < 100 mots | Réduire `max_seq_length` à 128 (gain de vitesse) |
| p95 > 400 mots | Augmenter `max_seq_length` à 512 (plus de mémoire) |
| Textes très hétérogènes | Garder 256 — bon compromis |

**Où ajuster** : `src/claims_classifier/config.py` → `max_seq_length`.

#### 4.3 Qualité des données

Questions à se poser :
- Y a-t-il des doublons ? → `df.duplicated("text").sum()`
- Y a-t-il des textes vides ou trop courts (< 5 mots) ? → filtrés automatiquement par `clean_dataframe()`
- Les labels sont-ils cohérents ? → vérifier `df["label"].value_counts()` après mapping
- Y a-t-il du HTML ou des caractères spéciaux récurrents ? → à traiter dans `clean_text()`

### Décisions selon ce qu'on observe

```
Moins de 5 000 exemples au total ?
  → Réduire embedding_dim à 64, cnn_num_filters à 64
  → Favoriser MLP (moins de paramètres, moins de surapprentissage)

Plus de 10 classes et déséquilibre fort ?
  → Vérifier que les poids de la loss dans losses.py sont bien calculés
  → Accepter F1 faible sur les classes rares (< 500 obs.)

Textes très courts (< 20 mots en médiane) ?
  → Réduire kernel_sizes à (2, 3, 4) dans config.py
  → Réduire max_seq_length à 64 ou 128
```

---

## 5. Configurer le projet

**Fichier central** : `src/claims_classifier/config.py`

Toute la configuration est centralisée ici et surchargeable par variable
d'environnement (format : `CLAIMS_TRAINING__BATCH_SIZE=128`).

### Hyperparamètres clés

#### Prétraitement (`PreprocessingConfig`)

| Paramètre | Défaut | Rôle | Quand l'ajuster |
|-----------|--------|------|-----------------|
| `vocab_size` | 30 000 | Mots retenus dans le vocabulaire | Réduire à 15 000 si peu de données ; augmenter à 40 000 pour le français |
| `max_seq_length` | 256 | Longueur max des séquences (en tokens) | Basé sur p95 des longueurs de texte |
| `min_word_frequency` | 2 | Fréquence minimale pour inclure un mot | Augmenter à 3-5 si beaucoup de bruit/typos |

#### Modèle (`ModelConfig`)

| Paramètre | Défaut | Rôle | Quand l'ajuster |
|-----------|--------|------|-----------------|
| `embedding_dim` | 128 | Dimension des vecteurs de mots | Réduire à 64 si < 5 000 exemples |
| `cnn_num_filters` | 128 | Filtres par taille de kernel | Réduire à 64 si surapprentissage |
| `cnn_kernel_sizes` | (3, 4, 5) | Tailles des n-grammes détectés | (2, 3, 4) pour textes très courts |
| `mlp_hidden_dim` | 64 | Couche cachée du MLP | Augmenter si les classes sont complexes |

#### Entraînement (`TrainingConfig`)

| Paramètre | Défaut | Rôle | Quand l'ajuster |
|-----------|--------|------|-----------------|
| `batch_size` | 64 | Taille des mini-batches | Réduire à 32 si OOM sur GPU ; augmenter à 128 sur grand dataset |
| `learning_rate` | 1e-3 | Vitesse d'apprentissage | Rarement à modifier — Adam s'adapte |
| `num_epochs` | 20 | Limite haute d'époques | L'early stopping s'arrêtera avant si nécessaire |
| `early_stopping_patience` | 3 | Époques sans amélioration avant arrêt | Augmenter à 5 si l'entraînement est bruité |

#### Monitoring (`MonitoringConfig`)

| Paramètre | Défaut | Quand l'ajuster |
|-----------|--------|-----------------|
| `low_confidence_threshold` | 0.5 | Seuil de confiance "faible" — adapter selon le SLA client |
| `class_drift_warning` | 0.10 | TVD > 10 % → alerte douce |
| `min_predictions_for_drift` | 50 | Réduire à 20 si le volume est faible en prod |

### Règles d'ajustement rapides

```
Peu de données (< 5 000 obs.) :
  vocab_size ↓15 000 · embedding_dim ↓64 · cnn_num_filters ↓64

Textes longs (p95 > 400 mots) :
  max_seq_length ↑512 · batch_size ↓32 (mémoire GPU)

Beaucoup de classes (> 15) :
  num_epochs ↑30 · early_stopping_patience ↑5

Dataset très équilibré :
  La loss pondérée est toujours valide — elle ne nuit pas
```

---

## 6. Entraîner

### Commandes

```bash
# Entraîner le TextCNN (recommandé par défaut)
.\tasks.ps1 train
# ou : uv run python scripts/train.py --model textcnn

# Entraîner le MLP (comparaison rapide)
uv run python scripts/train.py --model mlp

# Surcharger un hyperparamètre sans modifier config.py
$env:CLAIMS_TRAINING__BATCH_SIZE=32; uv run python scripts/train.py
```

**Référence** : `scripts/train.py` pour les options CLI complètes.

### MLP vs TextCNN : lequel choisir ?

| Contexte | Choisir |
|----------|---------|
| < 10 000 exemples | **MLP** — moins de paramètres, moins de surapprentissage |
| Textes très courts (< 30 mots) | **MLP** — le CNN perd de l'intérêt sur les courts contextes |
| > 10 000 exemples, textes longs | **TextCNN** — meilleur sur les motifs complexes |
| Comparaison rapide avec le client | MLP d'abord (< 5 min), puis TextCNN si insuffisant |

Dans ce projet : TextCNN gagne 2 points de F1 sur MLP. En général, l'écart est de 0 à 5 %.

### GPU vs CPU : où entraîner ?

| Volume | GPU local | CPU local | Cloud GPU |
|--------|-----------|-----------|-----------|
| < 10 000 obs. | 5 min | 30 min | Inutile |
| 50 000–100 000 obs. | 20–40 min | 4–6 h | Si pas de GPU local |
| > 200 000 obs. | 1–3 h | 1–2 jours | Google Colab Pro (10 €/mois) |

Le projet détecte automatiquement le GPU — aucune modification nécessaire.

### Lire les courbes d'entraînement

Lancer TensorBoard : `.\tasks.ps1 tensorboard` → `http://localhost:6006`

| Signe sur les courbes | Interprétation | Action |
|----------------------|----------------|--------|
| `val_loss` remonte alors que `train_loss` descend | Surapprentissage | Réduire `cnn_num_filters`, augmenter dropout |
| Les deux losses descendent ensemble | Normal, continue | — |
| `val_f1` plafonne dès l'époque 2-3 | Sous-apprentissage | Augmenter `embedding_dim`, vérifier les données |
| Early stopping déclenché à l'époque 4 | OK si val_f1 satisfaisant | Sinon, réduire le `learning_rate` |

---

## 7. Évaluer & valider

### Commandes

```bash
# Évaluation complète sur le jeu de test
.\tasks.ps1 evaluate
# Génère : reports/figures/confusion_matrix.png + rapport texte
```

**Référence** : `scripts/evaluate.py` et `src/claims_classifier/evaluation/metrics.py`.

### Quelle métrique selon le contexte client ?

| Contexte | Métrique principale | Pourquoi |
|----------|--------------------|-|
| Toutes les classes comptent pareil | **Macro F1** | Pénalise les classes oubliées |
| Les classes volumineuses dominent le coût | **Weighted F1** | Pondère par le volume de chaque classe |
| Un seul type d'erreur est inacceptable | **F1 par classe** | Identifier précisément quelle classe rate |
| Client veut minimiser les fausses alarmes | **Précision par classe** | Calculable depuis `evaluation/metrics.py` |

### Lire la matrice de confusion

**Fichier généré** : `reports/figures/confusion_matrix_*.png`

Ce qu'il faut chercher :
- **Diagonale** : prédictions correctes (plus c'est foncé, mieux c'est)
- **Hors diagonale** : confusions récurrentes → est-ce normal métier ?
  (ex: `debt_collection` confondu avec `credit_reporting` est acceptable —
  c'est souvent le même profil de client)
- **Ligne vide** : classe jamais prédite → trop rare dans le dataset
- **Colonne pleine** : classe sur-prédite → revoir le mapping ou la loss

### Définir le seuil d'acceptation avec le client

Exemples de critères :
- **Tri automatique d'e-mails** : Weighted F1 ≥ 75 % suffit (gain opérationnel immédiat)
- **Routage de tickets critiques** : F1 sur la classe critique ≥ 90 % obligatoire
- **Aide à la décision** : Macro F1 ≥ 60 % (on accepte que certaines classes soient floues)

> **Règle pratique** : demander au client "à partir de quel taux d'erreur cela
> coûte plus cher que le tri manuel ?" — c'est son seuil naturel.

---

## 8. Déployer

### 8.1 API FastAPI (mode développement / test)

```bash
.\tasks.ps1 api
# → http://localhost:8000/docs  (documentation interactive)
# → POST /predict               (prédiction)
# → GET  /health                (sonde de vie)
```

**Fichiers concernés** :
- `api/main.py` — application et configuration CORS
- `api/routers/predict.py` — logique des endpoints
- `api/schemas.py` — format des requêtes/réponses (à adapter si on ajoute des champs)

### 8.2 Docker (mode production)

```bash
docker build -t mon-classifieur-client .
docker run -p 8000:8000 mon-classifieur-client
```

L'image embarque le checkpoint `.pt` et les artefacts — elle est auto-suffisante.

**Référence** : `Dockerfile` à la racine. L'image est CPU-only (~1.4 Go).

### 8.3 Adapter pour la prod client

| Point d'adaptation | Fichier | Ce qu'on change |
|--------------------|---------|-----------------|
| Autoriser les appels depuis l'app web client | `api/main.py` | `allow_origins` dans `CORSMiddleware` — remplacer `["*"]` par le domaine client |
| Changer le port | `Dockerfile` / `tasks.ps1` | `--port 8080` par exemple |
| Authentification basique | `api/main.py` | Ajouter un middleware HTTP Basic ou Bearer token |
| Limiter le débit | `api/main.py` | Ajouter `slowapi` comme middleware de rate-limiting |
| HTTPS | Reverse proxy (Nginx, Traefik) | Devant le conteneur Docker — hors scope de ce template |

> **Pour une intégration CRM/ticketing** : le client appelle `POST /predict`
> avec le texte de la réclamation et reçoit la classe + la probabilité.
> Latence : ~10 ms sur CPU (acceptable pour tous les outils de ticketing).

---

## 9. Monitorer en production

### Mise en route du monitoring

```bash
# 1. Calculer la baseline (une seule fois après l'entraînement)
.\tasks.ps1 baseline        # nécessite complaints.csv + vocab.json

# 2. L'API logue automatiquement chaque prédiction dans logs/predictions.jsonl
.\tasks.ps1 api

# 3. Ouvrir le dashboard
.\tasks.ps1 dashboard       # → http://localhost:8501
```

**Fichiers concernés** :
- `src/claims_classifier/monitoring/logger.py` — format du log JSONL
- `src/claims_classifier/monitoring/baseline.py` — calcul des stats de référence
- `src/claims_classifier/monitoring/drift.py` — détection de dérive
- `monitoring/dashboard.py` — dashboard Streamlit

### Interpréter les indicateurs de dérive

| Indicateur | Seuil WARNING | Seuil ALERT | Cause probable |
|------------|--------------|-------------|----------------|
| **TVD classes** | > 10 % | > 20 % | Distribution des requêtes qui change (saison, événement) |
| **Longueur textes** | > 2× baseline | > 3× baseline | Nouveau canal d'entrée (e-mail vs SMS) |
| **Tokens inconnus** | > 15 % | > 30 % | Nouveau vocabulaire métier, jargon, langue |
| **Faible confiance** | > 20 % | > 40 % | Mélange de nouvelles classes non prévues |

### Quand ré-entraîner le modèle ?

```
TVD > 20 %  ET  volume de nouvelles données ≥ 500 exemples labellisés
  → Ré-entraîner avec les nouvelles données (ajouter au CSV d'origine)

Tokens inconnus > 30 %
  → Identifier les nouveaux mots, les ajouter au corpus, reconstruire le vocab

Faible confiance > 40 % sans TVD élevée
  → Vérifier si une nouvelle classe est apparue (non prévue dans LABEL_MAPPING)
  → Si oui : labelliser ~200 exemples et ajouter la classe

Performances stables depuis 3 mois
  → Ré-entraînement préventif recommandé si de nouvelles données sont disponibles
```

---

## 10. Checklist de réplication

À utiliser pour chaque nouvelle mission. Copier-coller dans l'outil de suivi du client.

### Préparation

- [ ] Cloner le template dans un dossier nommé `<client>-classifier/`
- [ ] Renommer le projet dans `pyproject.toml` (`name = "client-classifier"`)
- [ ] Installer les dépendances : `uv sync`
- [ ] Valider l'installation : `uv run pytest -m "not integration" -v` (12 tests verts)

### Adaptation du dataset

- [ ] Placer le CSV client dans `data/raw/` (nom : `complaints.csv` ou adapter `loader.py`)
- [ ] Vérifier les noms de colonnes : adapter `COL_TEXT` et `COL_LABEL` dans `loader.py`
- [ ] Lister tous les labels bruts du CSV (`df["label"].value_counts()`)
- [ ] Écrire `LABEL_MAPPING` dans `cleaning.py` (fusion → 5 à 20 classes)
- [ ] Valider le mapping : `df["label"].value_counts()` après application de `run_cleaning()`
- [ ] Supprimer ou adapter les regex inutiles dans `clean_text()`

### Analyse exploratoire

- [ ] Lancer `notebooks/01_eda.ipynb`
- [ ] Vérifier la distribution des classes (déséquilibre ?)
- [ ] Vérifier les longueurs de texte (p50, p75, p95)
- [ ] Ajuster `max_seq_length` dans `config.py` si nécessaire
- [ ] Identifier les classes < 100 obs. → décider : fusionner ou supprimer

### Configuration

- [ ] Ajuster `vocab_size` si dataset < 5 000 obs. (→ 15 000) ou langue française (→ 35 000)
- [ ] Ajuster `batch_size` selon la mémoire GPU disponible
- [ ] Définir avec le client le seuil F1 d'acceptation

### Entraînement

- [ ] Lancer MLP d'abord (entraînement rapide, benchmark) : `uv run python scripts/train.py --model mlp`
- [ ] Lancer TextCNN : `uv run python scripts/train.py --model textcnn`
- [ ] Vérifier les courbes TensorBoard — pas de surapprentissage
- [ ] Checkpoint sauvegardé dans `models/<nom>_best.pt`

### Évaluation

- [ ] Lancer `.\tasks.ps1 evaluate`
- [ ] Weighted F1 ≥ seuil accepté par le client ?
- [ ] Matrice de confusion : les confusions sont-elles métier-acceptables ?
- [ ] F1 par classe : aucune classe critique à zéro ?

### Déploiement

- [ ] Tester l'API en local : `.\tasks.ps1 api` → `POST http://localhost:8000/predict`
- [ ] Vérifier `/health` retourne `model_loaded: true`
- [ ] Builder l'image Docker : `docker build -t <client>-api .`
- [ ] Tester le conteneur : `docker run -p 8000:8000 <client>-api`
- [ ] Adapter `allow_origins` dans `api/main.py` (domaine client)

### Monitoring

- [ ] Générer la baseline : `.\tasks.ps1 baseline`
- [ ] Vérifier `data/processed/baseline_stats.json` créé
- [ ] Lancer le dashboard : `.\tasks.ps1 dashboard`
- [ ] Montrer le dashboard au client, définir les seuils de dérive
- [ ] Ajuster `MonitoringConfig` dans `config.py` si les seuils par défaut ne conviennent pas

---

## 11. Pièges connus & FAQ

### Classes ultra-rares (F1 = 0)

**Symptôme** : une classe n'est jamais prédite, F1 = 0 dans le rapport.

**Cause** : volume insuffisant (< 100 exemples) ou ratio de déséquilibre
extrême (> 500:1). La loss pondérée corrige partiellement mais pas totalement.

**Solution** : fusionner la classe dans `LABEL_MAPPING` vers `"other"` ou
la supprimer. Si la classe est critique métier, collecter plus de données
avant de ré-entraîner.

*Dans ce projet : classe `"other"` (187 obs., ratio 957:1) → F1 = 0.*

### Déséquilibre extrême

**Symptôme** : le modèle prédit toujours la classe majoritaire.

**Cause** : la loss pondérée n'est pas assez forte face à un déséquilibre > 200:1.

**Solution** :
1. Vérifier que `build_loss()` dans `training/losses.py` est bien utilisée
2. Augmenter artificiellement les exemples minoritaires (oversampling) avant
   `make_splits()` dans `dataset.py`
3. En dernier recours, supprimer la classe majoritaire jusqu'à un ratio ≤ 50:1

### Encodage Windows

**Symptôme** : `UnicodeEncodeError: 'charmap' codec can't encode character`

**Cause** : PowerShell en mode CP1252 ne supporte pas les emojis ou caractères
UTF-8 dans les f-strings.

**Solution** : ne pas utiliser d'emojis dans les fonctions `summary()` ou
les prints terminaux. Réserver les emojis aux interfaces Streamlit et Gradio
qui ont leur propre encodage.

*Ce bug a affecté `drift.py` — corrigé en utilisant `[OK]`, `[WARN]`, `[ALERT]`.*

### Fuite de données (Data Leakage)

**Symptôme** : les métriques de validation sont excellentes mais le modèle
performe mal sur de nouvelles données réelles.

**Cause** : le vocabulaire ou l'encodeur de labels a été construit sur tout
le dataset (train + val + test) au lieu du train seul.

**Règle inviolable** : `Vocabulary.build()` et `LabelEncoder.build()` ne
doivent recevoir que `train_df` — jamais `df` complet. Vérifié dans `dataset.py`
et dans le notebook `02_construction_modele.ipynb`.

### `torch.load()` warning sur les checkpoints

**Symptôme** : `FutureWarning: Weights only loading will be default in 2.x`

**Solution** : le projet utilise déjà `weights_only=True` partout où
`torch.load()` est appelé. Si le warning apparaît, vérifier que la version
de PyTorch est ≥ 2.5.

### Le CI échoue après ajout de nouveaux fichiers Python

**Cause** : ruff lint/format ne couvre pas les nouveaux dossiers.

**Solution** : mettre à jour `.github/workflows/ci.yml` pour inclure
le nouveau dossier dans les deux steps `ruff check` et `ruff format --check`.

### FAQ

**Q : Puis-je utiliser ce template sans GPU ?**
Oui. Désactiver CUDA dans `config.py` (`device = "cpu"`) ou laisser la
détection automatique. L'entraînement sera 5–20× plus lent selon le volume.

**Q : Combien de données minimum pour que ça fonctionne ?**
En pratique : 200 exemples minimum par classe, soit 2 000 exemples pour
10 classes. En dessous, les résultats seront décevants — envisager
un modèle pré-entraîné (BERT, CamemBERT) via HuggingFace Transformers.

**Q : Ce template peut-il gérer une classification multi-label ?**
Non — l'architecture actuelle est mono-label (une classe par texte).
Pour le multi-label, modifier la couche de sortie (`Sigmoid` au lieu de
`Softmax`) et la loss (`BCEWithLogitsLoss`). Changements dans `textcnn.py`,
`mlp.py` et `losses.py`.

**Q : Comment ajouter une classe après le déploiement ?**
Il faut ré-entraîner. Ajouter la nouvelle classe dans `LABEL_MAPPING`,
collecter ≥ 200 exemples labellisés, relancer `train.py`. Le checkpoint
sera écrasé — conserver l'ancien dans `models/` avec un nom versionné.

---

## 12. Estimation d'effort

Tableau réaliste pour une mission PME standard (données déjà collectées).

| Étape | Durée estimée | Dépend de… |
|-------|---------------|-----------|
| Prise en main du template + installation | 1–2 h | Familiarité avec uv, Python |
| Analyse du CSV client + mapping des classes | 2–4 h | Qualité des labels, discussions métier |
| Adaptation `cleaning.py` + `loader.py` | 1–2 h | Spécificités du domaine |
| EDA (notebook) + décisions config | 2–3 h | Hétérogénéité des données |
| Premier entraînement MLP (benchmark) | 30 min – 2 h | Volume de données, GPU dispo |
| Entraînement TextCNN (modèle final) | 30 min – 4 h | Idem |
| Évaluation + restitution client | 1–2 h | Nombre d'itérations |
| Déploiement API + Docker | 1–2 h | Infrastructure client |
| Configuration du monitoring | 1 h | — |
| Formation du client au dashboard | 1–2 h | Niveau technique du client |
| **TOTAL** | **10–22 h** | Soit 1,5 à 3 jours-consultant |

> **Note** : le goulot d'étranglement est presque toujours la **qualité et
> la labellisation des données client**, pas le code. Prévoir 1 atelier
> de 2–3 h avec le client pour aligner sur les classes finales avant de
> commencer à coder.

---

*Manuel rédigé par Christophe TROËL — TROËL Operations & Consulting*
*Stabiliser le quotidien. Automatiser le futur.*
