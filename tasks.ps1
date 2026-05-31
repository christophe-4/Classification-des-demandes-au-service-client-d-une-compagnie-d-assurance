# tasks.ps1 — Commandes courantes du projet
# Usage : .\tasks.ps1 <commande>
# Exemple : .\tasks.ps1 install

param (
    [Parameter(Mandatory=$true)]
    [string]$Task
)

switch ($Task) {

    "install" {
        Write-Host "Installation des dependances..." -ForegroundColor Cyan
        uv sync --all-groups
    }

    "lint" {
        Write-Host "Linting avec ruff..." -ForegroundColor Cyan
        uv run ruff check src/ tests/ scripts/
    }

    "format" {
        Write-Host "Formatage avec ruff..." -ForegroundColor Cyan
        uv run ruff format src/ tests/ scripts/
    }

    "test" {
        Write-Host "Lancement des tests..." -ForegroundColor Cyan
        uv run pytest tests/ -v
    }

    "train" {
        Write-Host "Lancement de l'entrainement..." -ForegroundColor Cyan
        uv run python scripts/train.py
    }

    "evaluate" {
        Write-Host "Evaluation du modele..." -ForegroundColor Cyan
        uv run python scripts/evaluate.py
    }

    "api" {
        Write-Host "Lancement de l'API FastAPI..." -ForegroundColor Cyan
        uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    }

    "dashboard" {
        Write-Host "Lancement du dashboard Streamlit..." -ForegroundColor Cyan
        uv run streamlit run monitoring/dashboard.py
    }

    "baseline" {
        Write-Host "Calcul de la baseline de reference..." -ForegroundColor Cyan
        uv run python scripts/compute_baseline.py
    }

    "tensorboard" {
        Write-Host "Lancement de TensorBoard..." -ForegroundColor Cyan
        uv run tensorboard --logdir runs/
    }

    "clean" {
        Write-Host "Nettoyage des fichiers temporaires..." -ForegroundColor Cyan
        Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
        Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
        Write-Host "Nettoyage termine." -ForegroundColor Green
    }

    "check" {
        Write-Host "Verification de l'environnement..." -ForegroundColor Cyan
        uv run python -c "import torch; print('PyTorch :', torch.__version__)"
        uv run python -c "import torch; print('CUDA :', torch.cuda.is_available())"
        uv run python -c "import pandas; print('Pandas :', pandas.__version__)"
        uv run python -c "import sklearn; print('Scikit-learn :', sklearn.__version__)"
    }

    default {
        Write-Host "Commandes disponibles :" -ForegroundColor Yellow
        Write-Host "  install     - Installer les dependances"
        Write-Host "  lint        - Verifier le code avec ruff"
        Write-Host "  format      - Formater le code avec ruff"
        Write-Host "  test        - Lancer les tests pytest"
        Write-Host "  train       - Lancer l'entrainement"
        Write-Host "  evaluate    - Evaluer le modele"
        Write-Host "  api         - Lancer l'API FastAPI (port 8000)"
        Write-Host "  dashboard   - Lancer le dashboard Streamlit de monitoring"
        Write-Host "  baseline    - Calculer la baseline de reference (necessite complaints.csv)"
        Write-Host "  tensorboard - Lancer TensorBoard"
        Write-Host "  clean       - Nettoyer les fichiers temporaires"
        Write-Host "  check       - Verifier l'environnement"
    }
}