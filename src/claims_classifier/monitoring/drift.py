# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Detection de derive (data drift) en production.

Compare les statistiques des predictions recentes avec la baseline
d'entrainement pour identifier :
  1. Derive de la distribution des classes (TVD)
  2. Derive des longueurs de texte
  3. Taux de tokens inconnus eleve (nouveau vocabulaire)
  4. Taux de predictions a faible confiance

Statuts possibles par indicateur :
  - OK      : dans les seuils normaux
  - WARNING : derive moderee, a surveiller
  - ALERT   : derive significative, action recommandee

Les seuils sont configures dans config.py (MonitoringConfig).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from claims_classifier.config import config
from claims_classifier.monitoring.logger import load_logs

logger = logging.getLogger(__name__)


# =============================================================================
# TYPES
# =============================================================================


class DriftStatus(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    ALERT = "ALERT"


@dataclass
class DriftIndicator:
    """Resultat d'un indicateur de derive."""

    name: str
    status: DriftStatus
    value: float
    threshold_warning: float
    threshold_alert: float
    message: str
    unit: str = ""


@dataclass
class DriftReport:
    """Rapport complet de detection de derive."""

    computed_at: str
    num_predictions_analyzed: int
    window_description: str
    indicators: list[DriftIndicator] = field(default_factory=list)
    overall_status: DriftStatus = DriftStatus.OK

    def summary(self) -> str:
        """Retourne un resume lisible du rapport (ASCII-safe)."""
        lines = [
            f"Rapport de derive -- {self.computed_at}",
            f"Predictions analysees : {self.num_predictions_analyzed} ({self.window_description})",
            f"Statut global         : {self.overall_status.value}",
            "-" * 60,
        ]
        for ind in self.indicators:
            icon = {"OK": "[OK]", "WARNING": "[WARN]", "ALERT": "[ALERT]"}.get(ind.status.value, "[?]")
            lines.append(
                f"{icon:<8} {ind.name:<40} {ind.value:.4f}{ind.unit}"
                f"  -- {ind.message}"
            )
        return "\n".join(lines)


# =============================================================================
# CALCULS INTERNES
# =============================================================================


def _status_from_value(value: float, warn: float, alert: float) -> DriftStatus:
    """Retourne le statut selon les seuils."""
    if value >= alert:
        return DriftStatus.ALERT
    if value >= warn:
        return DriftStatus.WARNING
    return DriftStatus.OK


def _total_variation_distance(observed: dict, baseline: dict) -> float:
    """
    Calcule la Total Variation Distance entre deux distributions.

    TVD = 0.5 * sum(|p_i - q_i|)
    Valeur : 0 (distributions identiques) a 1 (distributions completement differentes).

    Args:
        observed : Distribution observee {class: proportion}.
        baseline : Distribution de reference {class: proportion}.

    Returns:
        TVD (float entre 0 et 1).
    """
    all_classes = set(baseline.keys()) | set(observed.keys())
    tvd = 0.5 * sum(abs(observed.get(cls, 0.0) - baseline.get(cls, 0.0)) for cls in all_classes)
    return round(tvd, 6)


# =============================================================================
# DETECTION DE DERIVE
# =============================================================================


def detect_drift(
    logs: list[dict] | None = None,
    baseline: dict | None = None,
    last_n: int = 500,
) -> DriftReport | None:
    """
    Compare les predictions recentes a la baseline et retourne un rapport de derive.

    Args:
        logs     : Liste de predictions (depuis load_logs). Si None, charge automatiquement.
        baseline : Statistiques de reference. Si None, charge depuis baseline_stats.json.
        last_n   : Nombre de predictions recentes a analyser.

    Returns:
        DriftReport avec un statut par indicateur, ou None si les donnees sont insuffisantes.
    """
    from datetime import datetime, timezone

    from claims_classifier.monitoring.baseline import load_baseline

    # ── Chargement des donnees ────────────────────────────────────────────────
    if logs is None:
        logs = load_logs(n=last_n)
    else:
        logs = logs[-last_n:]

    if baseline is None:
        baseline = load_baseline()

    cfg = config.monitoring
    min_pred = cfg.min_predictions_for_drift

    if len(logs) < min_pred:
        logger.warning(
            f"Seulement {len(logs)} predictions disponibles "
            f"(minimum requis : {min_pred}). Derive non calculable."
        )
        return None

    if baseline is None:
        logger.warning("Baseline introuvable. Derive non calculable.")
        return None

    now = datetime.now(timezone.utc).isoformat()
    indicators: list[DriftIndicator] = []

    # ── 1. Derive de la distribution des classes ──────────────────────────────
    pred_class_counts: dict[str, int] = {}
    for rec in logs:
        cls = rec.get("predicted_class", "unknown")
        pred_class_counts[cls] = pred_class_counts.get(cls, 0) + 1

    total = len(logs)
    observed_distribution = {cls: count / total for cls, count in pred_class_counts.items()}
    baseline_distribution = baseline.get("class_distribution", {})

    tvd = _total_variation_distance(observed_distribution, baseline_distribution)
    status_tvd = _status_from_value(tvd, cfg.class_drift_warning, cfg.class_drift_alert)

    top_obs = max(observed_distribution, key=observed_distribution.get) if observed_distribution else "?"
    top_base = max(baseline_distribution, key=baseline_distribution.get) if baseline_distribution else "?"
    indicators.append(
        DriftIndicator(
            name="Distribution des classes (TVD)",
            status=status_tvd,
            value=tvd,
            threshold_warning=cfg.class_drift_warning,
            threshold_alert=cfg.class_drift_alert,
            message=f"Classe dominante : obs={top_obs} vs baseline={top_base}",
            unit="",
        )
    )

    # ── 2. Longueur des textes ────────────────────────────────────────────────
    lengths = [rec.get("text_length", 0) for rec in logs]
    median_obs = sorted(lengths)[len(lengths) // 2]
    median_base = baseline.get("text_length_stats", {}).get("median", median_obs)

    ratio = max(median_obs, median_base) / max(min(median_obs, median_base), 1)
    status_len = _status_from_value(ratio, cfg.text_length_ratio_warning, cfg.text_length_ratio_alert)

    indicators.append(
        DriftIndicator(
            name="Longueur des textes (ratio median)",
            status=status_len,
            value=ratio,
            threshold_warning=cfg.text_length_ratio_warning,
            threshold_alert=cfg.text_length_ratio_alert,
            message=f"Median obs={median_obs:.0f} mots vs baseline={median_base:.0f} mots",
            unit="x",
        )
    )

    # ── 3. Taux de tokens inconnus ────────────────────────────────────────────
    unk_rates = [rec.get("unk_rate", 0.0) for rec in logs]
    mean_unk = sum(unk_rates) / len(unk_rates)
    baseline_unk_mean = baseline.get("unk_rate_stats", {}).get("mean", 0.0)
    status_unk = _status_from_value(mean_unk, cfg.unk_rate_warning, cfg.unk_rate_alert)

    indicators.append(
        DriftIndicator(
            name="Taux tokens inconnus (mean)",
            status=status_unk,
            value=mean_unk,
            threshold_warning=cfg.unk_rate_warning,
            threshold_alert=cfg.unk_rate_alert,
            message=f"Mean obs={mean_unk:.4f} vs baseline={baseline_unk_mean:.4f}",
            unit="",
        )
    )

    # ── 4. Taux de predictions a faible confiance ────────────────────────────
    confidences = [rec.get("confidence", 1.0) for rec in logs]
    low_conf_rate = sum(1 for c in confidences if c < cfg.low_confidence_threshold) / len(confidences)
    status_conf = _status_from_value(
        low_conf_rate, cfg.low_confidence_rate_warning, cfg.low_confidence_rate_alert
    )
    mean_conf = sum(confidences) / len(confidences)

    indicators.append(
        DriftIndicator(
            name="Taux faible confiance (< 0.5)",
            status=status_conf,
            value=low_conf_rate,
            threshold_warning=cfg.low_confidence_rate_warning,
            threshold_alert=cfg.low_confidence_rate_alert,
            message=f"{low_conf_rate * 100:.1f}% de predictions incertaines | confiance moyenne={mean_conf:.4f}",
            unit="",
        )
    )

    # ── Statut global = pire statut individuel ────────────────────────────────
    priority = {DriftStatus.OK: 0, DriftStatus.WARNING: 1, DriftStatus.ALERT: 2}
    overall = max(indicators, key=lambda ind: priority[ind.status]).status

    return DriftReport(
        computed_at=now,
        num_predictions_analyzed=total,
        window_description=f"{last_n} dernieres predictions",
        indicators=indicators,
        overall_status=overall,
    )
