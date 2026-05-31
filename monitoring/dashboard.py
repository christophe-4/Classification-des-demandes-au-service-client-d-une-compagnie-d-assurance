# Copyright 2026 Christophe TROËL
# SPDX-License-Identifier: Apache-2.0

"""
Dashboard Streamlit de monitoring — Claims Classifier.

Lance avec :
    uv run streamlit run monitoring/dashboard.py

Lit :
  - logs/predictions.jsonl          (predictions en production)
  - data/processed/baseline_stats.json  (reference d'entrainement)
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permet d'importer depuis src/ sans installation en mode editable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claims_classifier.monitoring.baseline import load_baseline
from claims_classifier.monitoring.drift import DriftStatus, detect_drift
from claims_classifier.monitoring.logger import load_logs

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="Claims Classifier — Monitoring",
    page_icon="🏦",
    layout="wide",
)

# =============================================================================
# CHARGEMENT DES DONNEES
# =============================================================================


@st.cache_data(ttl=30)
def get_logs() -> list[dict]:
    return load_logs()


@st.cache_data(ttl=300)
def get_baseline() -> dict | None:
    return load_baseline()


# =============================================================================
# EN-TETE
# =============================================================================

st.title("🏦 Claims Classifier — Monitoring Production")
st.caption("TextCNN from scratch · Weighted F1 = 83.12 % · 12 classes financières")

# Bouton de rafraichissement (force un nouveau chargement)
col_refresh, col_info = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Rafraîchir"):
        st.cache_data.clear()
        st.rerun()

logs = get_logs()
baseline = get_baseline()

# =============================================================================
# STATUT GLOBAL
# =============================================================================

if not logs:
    st.info("📭 Aucune prédiction enregistrée. Lancez l'API et envoyez des requêtes.")
    if baseline is None:
        st.warning("⚠️ Baseline introuvable. Lancez : `uv run python scripts/compute_baseline.py`")
    st.stop()

# Calcul du rapport de derive
drift_report = None
if baseline and len(logs) >= 50:
    drift_report = detect_drift(logs=logs, baseline=baseline)

# Bandeau de statut global
if drift_report is None:
    status_color, status_icon, status_text = (
        "blue",
        "🔵",
        "Données insuffisantes (< 50 prédictions)",
    )
elif drift_report.overall_status == DriftStatus.OK:
    status_color, status_icon, status_text = "green", "🟢", "Modèle sain"
elif drift_report.overall_status == DriftStatus.WARNING:
    status_color, status_icon, status_text = "orange", "🟡", "Attention — dérive détectée"
else:
    status_color, status_icon, status_text = "red", "🔴", "ALERTE — dérive significative"

_bg = {"green": "d4edda", "orange": "fff3cd", "red": "f8d7da", "blue": "d1ecf1"}[status_color]
st.markdown(
    f"<div style='background-color:#{_bg};"
    f"padding:12px;border-radius:6px;font-size:1.1em;font-weight:bold;'>"
    f"{status_icon} Statut global : {status_text}"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown("")

# =============================================================================
# KPIs EN HAUT
# =============================================================================

df = pd.DataFrame(logs)

total_preds = len(df)
mean_confidence = df["confidence"].mean()
mean_latency = df["inference_time_ms"].mean()
low_conf_rate = (df["confidence"] < 0.5).mean() * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("📊 Prédictions totales", f"{total_preds:,}")
k2.metric("🎯 Confiance moyenne", f"{mean_confidence:.1%}")
k3.metric("⚡ Latence moyenne", f"{mean_latency:.1f} ms")
k4.metric("⚠️ Faible confiance", f"{low_conf_rate:.1f}%", help="Prédictions avec confiance < 50%")

st.divider()

# =============================================================================
# SECTION 1 — VOLUME DANS LE TEMPS
# =============================================================================

st.subheader("📈 Volume de prédictions dans le temps")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df_time = df.set_index("timestamp").resample("1h").size().reset_index(name="count")
df_time.columns = ["Heure", "Prédictions"]

if len(df_time) > 1:
    st.line_chart(df_time.set_index("Heure"))
else:
    st.bar_chart(df_time.set_index("Heure"))

# =============================================================================
# SECTION 2 — DISTRIBUTION DES CLASSES (LE GRAPHIQUE DE DERIVE)
# =============================================================================

st.subheader("🔍 Distribution des classes — Prédit vs Entraînement")

if baseline:
    class_dist_obs = df["predicted_class"].value_counts(normalize=True).rename("Prédit (%)")
    class_dist_base = pd.Series(baseline["class_distribution"]).rename("Baseline (%)")

    comparison_df = pd.DataFrame(
        {"Prédit (%)": class_dist_obs, "Baseline (%)": class_dist_base}
    ).fillna(0)
    comparison_df = comparison_df.sort_values("Baseline (%)", ascending=False)

    st.bar_chart(comparison_df)

    if drift_report:
        tvd_ind = next((i for i in drift_report.indicators if "TVD" in i.name), None)
        if tvd_ind:
            color = {"OK": "green", "WARNING": "orange", "ALERT": "red"}[tvd_ind.status.value]
            st.markdown(
                f"<span style='color:{color};font-weight:bold;'>"
                f"TVD = {tvd_ind.value:.4f} [{tvd_ind.status.value}] — {tvd_ind.message}"
                f"</span>",
                unsafe_allow_html=True,
            )
else:
    st.warning("Baseline non disponible — graphique comparatif impossible.")
    class_counts = df["predicted_class"].value_counts()
    st.bar_chart(class_counts)

st.divider()

# =============================================================================
# SECTION 3 — DISTRIBUTION DE LA CONFIANCE
# =============================================================================

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🎯 Distribution de la confiance")
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels = ["[0-0.3)", "[0.3-0.5)", "[0.5-0.7)", "[0.7-0.9)", "[0.9-1.0]"]
    df["conf_bin"] = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
    conf_dist = df["conf_bin"].value_counts().reindex(labels).fillna(0)
    st.bar_chart(conf_dist)

with col_b:
    st.subheader("📏 Longueur des textes (mots)")
    length_bins = [0, 20, 50, 100, 200, 500, 10000]
    length_labels = ["0-20", "21-50", "51-100", "101-200", "201-500", "500+"]
    df["length_bin"] = pd.cut(df["text_length"], bins=length_bins, labels=length_labels)
    length_dist = df["length_bin"].value_counts().reindex(length_labels).fillna(0)
    st.bar_chart(length_dist)

st.divider()

# =============================================================================
# SECTION 4 — RAPPORT DE DERIVE
# =============================================================================

st.subheader("🚨 Détection de dérive")

if drift_report is None:
    st.info(f"Rapport disponible dès {50} prédictions. Actuellement : {total_preds} prédiction(s).")
    if baseline is None:
        st.warning("Baseline introuvable. Lancez : `uv run python scripts/compute_baseline.py`")
else:
    st.caption(
        f"Basé sur {drift_report.num_predictions_analyzed} prédictions · "
        f"Calculé le {drift_report.computed_at[:19].replace('T', ' ')} UTC"
    )

    for ind in drift_report.indicators:
        status_colors = {"OK": "🟢", "WARNING": "🟡", "ALERT": "🔴"}
        icon = status_colors[ind.status.value]
        bg = {"OK": "#d4edda", "WARNING": "#fff3cd", "ALERT": "#f8d7da"}[ind.status.value]

        st.markdown(
            f"<div style='background:{bg};padding:10px;border-radius:5px;margin:4px 0;'>"
            f"<strong>{icon} {ind.name}</strong>&nbsp;&nbsp;"
            f"valeur = <code>{ind.value:.4f}{ind.unit}</code> &nbsp;|&nbsp; "
            f"seuils : warning={ind.threshold_warning}, alert={ind.threshold_alert}<br/>"
            f"<em>{ind.message}</em>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# =============================================================================
# SECTION 5 — DERNIÈRES PRÉDICTIONS (RGPD — sans texte brut)
# =============================================================================

st.subheader("🗂️ 20 dernières prédictions")
st.caption("⚠️ RGPD : aucun texte brut — uniquement des métadonnées dérivées")

cols_display = [
    "timestamp",
    "predicted_class",
    "confidence",
    "text_length",
    "unk_rate",
    "inference_time_ms",
]
display_df = df[cols_display].tail(20).sort_index(ascending=False).copy()
display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
display_df["confidence"] = display_df["confidence"].map("{:.1%}".format)
display_df["unk_rate"] = display_df["unk_rate"].map("{:.1%}".format)
display_df["inference_time_ms"] = display_df["inference_time_ms"].map("{:.1f} ms".format)
display_df.columns = [
    "Horodatage",
    "Classe prédite",
    "Confiance",
    "Longueur (mots)",
    "Taux <unk>",
    "Latence",
]

st.dataframe(display_df, use_container_width=True, hide_index=True)

# =============================================================================
# PIED DE PAGE
# =============================================================================

st.caption(f"📁 Source : `{Path('logs/predictions.jsonl')}` · {total_preds:,} prédictions au total")
