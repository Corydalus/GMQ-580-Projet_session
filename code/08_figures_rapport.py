"""08 — Figures de synthèse du rapport & de la présentation orale.

Figures transversales aux deux méthodes (avec / sans élévation), construites à partir
des rapports de métriques versionnés (`outputs/*/logs/03_model_rapport.json`,
`04_predict_rapport.json`, `commun/logs/02_ebird_rapport.json`) et — pour les courbes
ROC — d'une reproduction de la validation croisée spatiale (`random_state=42`).

Sorties :
- comparaison/figures/comparaison_performances.png   (A1 — AUC + TSS des 4 configs)
- comparaison/figures/importance_combine.png         (A2 — dominance de la détection)
- comparaison/figures/roc_folds.png                  (A4 — ROC par fold, re-run CV)
- comparaison/figures/tableau_metriques.png          (A3 — synthèse des métriques)
- commun/figures/entonnoir_ebird.png                 (B4 — cascade de filtrage eBird)
- comparaison/figures/pipeline.png                   (B1/B2 — pipeline + sources→variables)

Usage : `uv run python code/08_figures_rapport.py`
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "code"))
log = logging.getLogger("08_figures_rapport")

# ── Palette colorblind-safe (Okabe-Ito) — assignée par entité, jamais recyclée ──
C_AVEC, C_SANS = "#0072B2", "#D55E00"          # avec / sans élévation
C_DETECT, C_HABITAT = "#D55E00", "#0072B2"     # variables de détection / d'habitat
C_PRES, C_ABS = "#D55E00", "#B7C4CE"           # présence / absence
INK, MUTED, GRID = "#1a1a1a", "#5f5f5f", "#e8e8e8"

ETIQUETTES = {
    "MHC_hauteur_canopee": "Hauteur de canopée", "TWI": "TWI",
    "densite_lisiere": "Densité de lisière", "prop_feuillu_melange": "Prop. feuillu/mélangé",
    "densite_routes": "Densité de routes", "LST_estivale": "LST estivale",
    "classe_age": "Classe d'âge", "densite_peuplement": "Densité de peuplement",
    "distance_milieu_humide": "Distance milieu humide", "elevation": "Élévation",
    "minutes_apres_coucher": "Minutes après coucher", "phase_lune": "Phase lunaire",
    "log_duree": "log(durée liste)", "jour_julien": "Jour julien",
}
DETECTION = {"minutes_apres_coucher", "phase_lune", "log_duree", "jour_julien"}


def _lire(rel: str) -> dict:
    return json.loads((RACINE / rel).read_text())


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
                         "axes.titlesize": 11, "figure.dpi": 150})


def _sans_cadre(ax, garder=("left", "bottom")) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in garder)


# ── A1 — Comparaison de performance (AUC + TSS, 4 configs) ────────────────────

def fig_performances(avec: dict, sans: dict, sortie: Path) -> None:
    """Barres groupées AUC-ROC + TSS — valeurs au-dessus des barres d'erreur, légende hors zone."""
    groupes = ["Modèle combiné\n(habitat + détection)", "Modèle habitat\n(H1–H4)"]
    x, w = np.arange(len(groupes)), 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9))
    specs = [(axes[0], "auc", "AUC-ROC", 0.5, 1.10), (axes[1], "tss", "TSS", 0.0, 1.12)]
    handles = None
    for ax, cle, titre, ymin, ymax in specs:
        for src, coul, lab, dx in [(avec, C_AVEC, "Avec élévation", -w / 2),
                                   (sans, C_SANS, "Sans élévation", w / 2)]:
            vals = [src["modele_combine"][f"{cle}_moy"], src["modele_habitat"][f"{cle}_moy"]]
            errs = [src["modele_combine"][f"{cle}_std"], src["modele_habitat"][f"{cle}_std"]]
            bars = ax.bar(x + dx, vals, w, yerr=errs, capsize=4, color=coul, label=lab,
                          error_kw={"elinewidth": 1.1, "ecolor": MUTED})
            for b, v, e in zip(bars, vals, errs):        # valeur AU-DESSUS de la moustache
                ax.text(b.get_x() + b.get_width() / 2, v + e + (ymax - ymin) * 0.018, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
        handles = ax.get_legend_handles_labels()[0]
        ax.set_xticks(x)
        ax.set_xticklabels(groupes, fontsize=9)
        ax.set_ylim(ymin, ymax)
        ax.set_title(titre, fontsize=10.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        _sans_cadre(ax)
    axes[0].set_ylabel("Score (moyenne ± écart-type inter-folds)")
    fig.legend(handles, ["Avec élévation", "Sans élévation"], loc="lower center",
               ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Puissance discriminante des modèles — effet du retrait de l'élévation",
                 fontsize=12.5, y=0.99)
    fig.text(0.5, 0.905, "Validation croisée spatiale · blocs de 10 km · 5 folds",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    fig.savefig(sortie, bbox_inches="tight")
    plt.close(fig)
    log.info("  écrit %s", sortie.name)


# ── A2 — Importance du modèle combiné : la détection domine ───────────────────

def fig_importance_combine(avec: dict, sortie: Path) -> None:
    """Importance par permutation du modèle combiné (14 var) — détection vs habitat."""
    imp = sorted(avec["modele_combine"]["importance"], key=lambda d: d["importance"])
    noms = [ETIQUETTES.get(d["variable"], d["variable"]) for d in imp]
    vals = [d["importance"] for d in imp]
    coul = [C_DETECT if d["variable"] in DETECTION else C_HABITAT for d in imp]
    y = np.arange(len(imp))
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.barh(y, vals, color=coul, height=0.72)
    vmax = max(vals)
    for yi, v in zip(y, vals):                             # valeur à droite de la barre
        ax.text(v + vmax * 0.012, yi, f"{v:.4f}", va="center", ha="left",
                fontsize=8.3, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(noms, fontsize=9)
    ax.set_xlim(0, vmax * 1.16)
    ax.set_xlabel("Importance par permutation (chute d'AUC-ROC)")
    ax.set_title("Modèle combiné : la variable de détection écrase toutes les autres",
                 fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    _sans_cadre(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_DETECT),
               plt.Rectangle((0, 0), 1, 1, color=C_HABITAT)]
    ax.legend(handles, ["Variable de détection", "Variable d'habitat"],
              loc="lower right", frameon=False, fontsize=9.5)
    fig.text(0.5, -0.02, "→ justifie l'entraînement d'un modèle « habitat seul » "
             "pour interpréter les hypothèses H1–H4", ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(sortie, bbox_inches="tight")
    plt.close(fig)
    log.info("  écrit %s", sortie.name)


# ── A3 — Tableau de synthèse des métriques ────────────────────────────────────

def _params_txt(p: dict) -> str:
    mf = p["max_features"]
    mf = f"{mf}" if isinstance(mf, str) else f"{mf:g}"
    md = p["max_depth"] if p["max_depth"] is not None else "∞"
    return f"n={p['n_estimators']} · prof={md} · feat={mf} · feuille={p['min_samples_leaf']}"


def fig_tableau_metriques(avec: dict, sans: dict, pred_a: dict, pred_s: dict,
                          ebird: dict, sortie: Path) -> None:
    """Tableau PNG : AUC/TSS ±σ, hyperparamètres, seuil — 4 configurations."""
    lignes = []
    for src, pred, lab in [(avec, pred_a, "avec élévation"), (sans, pred_s, "sans élévation")]:
        c, h = src["modele_combine"], src["modele_habitat"]
        lignes.append(["Combiné (carte)", lab, str(len(c["features"])),
                       f"{c['auc_moy']:.3f} ± {c['auc_std']:.3f}",
                       f"{c['tss_moy']:.3f} ± {c['tss_std']:.3f}",
                       _params_txt(c["best_params"]), f"{pred['seuil_proba']:.3f}"])
        lignes.append(["Habitat (H1–H4)", lab, str(len(h["features"])),
                       f"{h['auc_moy']:.3f} ± {h['auc_std']:.3f}",
                       f"{h['tss_moy']:.3f} ± {h['tss_std']:.3f}",
                       _params_txt(h["best_params"]), "—"])
    entetes = ["Modèle", "Élévation", "Var.", "AUC-ROC", "TSS", "Hyperparamètres RF", "Seuil"]
    fig, ax = plt.subplots(figsize=(12.2, 2.5))
    ax.axis("off")
    tab = ax.table(cellText=lignes, colLabels=entetes, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.55)
    ncol = len(entetes)
    for (r, cc), cell in tab.get_celld().items():
        cell.set_edgecolor("#d5d5d5")
        if r == 0:
            cell.set_facecolor("#33475b")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#eef3f7" if r % 2 else "white")
            if r == 1:                                    # ligne du modèle-carte retenu
                cell.set_facecolor("#dcebf7")
        if cc in (5,):
            cell.set_text_props(fontsize=8, ha="left")
            cell._loc = "left"
    tab.auto_set_column_width(col=list(range(ncol)))
    ax.set_title(f"Synthèse des performances — {ebird['n_final']:,} checklists · "
                 f"{ebird['n_presences']} présences ({ebird['ratio_presence']*100:.2f} %) · "
                 f"années {ebird['annee_min']}–{ebird['annee_max']}".replace(",", " "),
                 fontsize=11, pad=14)
    fig.savefig(sortie, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("  écrit %s", sortie.name)


# ── B4 — Entonnoir d'effort eBird ─────────────────────────────────────────────

def fig_entonnoir(ebird: dict, sortie: Path) -> None:
    """Cascade de filtrage (largeur ∝ log10 n) + rareté présence/absence."""
    n_tot = ebird["n_checklists_total"]
    n_filt = ebird["n_apres_filtres"]
    n_fin = ebird["n_final"]
    n_pres, n_abs = ebird["n_presences"], ebird["n_absences"]
    etapes = [("Checklists eBird (EBD)", n_tot, 100.0),
              ("Filtres d'effort (Johnston 2021)", n_filt, 100 * n_filt / n_tot),
              ("Dans l'emprise + covariables", n_fin, 100 * n_fin / n_tot)]

    fig, (ax, axb) = plt.subplots(2, 1, figsize=(8.6, 5.4),
                                  gridspec_kw={"height_ratios": [3.4, 1]})
    lw = [np.log10(n) for _, n, _ in etapes]
    for i, ((lab, n, pct), w) in enumerate(zip(etapes, lw)):
        yb = (len(etapes) - 1 - i) * 1.15
        ax.barh(yb, w, left=-w / 2, height=0.52, color=C_AVEC, alpha=0.35 + 0.22 * i)
        ax.text(0, yb + 0.07, f"{n:,}".replace(",", " "), ha="center", va="center",
                fontsize=13, weight="bold", color=INK)
        ax.text(0, yb - 0.42, f"{lab}   ({pct:.1f} % du total)", ha="center", va="center",
                fontsize=9, color=MUTED)
    ax.set_xlim(-max(lw) * 0.64, max(lw) * 0.64)
    ax.set_ylim(-0.75, (len(etapes) - 1) * 1.15 + 0.55)
    ax.axis("off")
    ax.set_title("De 2,9 millions de checklists à la table modèle", fontsize=11.5, pad=8)

    # Rareté : les 17 440 checklists = 148 présences (sliver) + 17 292 absences.
    axb.barh(0, n_abs, left=n_pres, color=C_ABS, height=0.42, label="Absences")
    axb.barh(0, n_pres, color=C_PRES, height=0.42, label="Présences")
    axb.set_xlim(0, n_fin)
    axb.set_ylim(-0.7, 0.95)
    axb.axis("off")
    axb.annotate(f"{n_pres} présences ({100*n_pres/n_fin:.2f} %)",
                 xy=(n_pres, 0.21), xytext=(n_fin * 0.10, 0.72), fontsize=9.5, color=C_PRES,
                 weight="bold", arrowprops=dict(arrowstyle="-", color=C_PRES, lw=0.9))
    axb.text(n_fin * 0.55, 0, f"{n_abs:,} absences confirmées (zero-fill)".replace(",", " "),
             ha="center", va="center", fontsize=9, color="#37474f")
    axb.text(0, -0.62, "Fort déséquilibre des classes (ratio 1:117)", ha="left",
             va="center", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(h_pad=1.8)
    fig.savefig(sortie, bbox_inches="tight")
    plt.close(fig)
    log.info("  écrit %s", sortie.name)


# ── A4 — Courbes ROC par fold (reproduction de la CV spatiale) ────────────────

def _module_modele():
    """Charge 03_model.py comme module (nom de fichier non importable directement)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("modele03", RACINE / "code/03_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _roc_moyenne(m, cfg, features: list[str], params: dict):
    """ROC interpolée moyenne ± σ sur les 5 folds spatiaux (mêmes blocs, random_state=42)."""
    from sklearn.metrics import auc as _auc, roc_curve
    X, y, groups, _ = m.charger_donnees(cfg, features)
    cv = m.make_cv(cfg)
    grille = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    for itr, ite in cv.split(X, y, groups):
        mod = m._rf(cfg, n_jobs=-1, **params).fit(X[itr], y[itr])
        p = mod.predict_proba(X[ite])[:, 1]
        fpr, tpr, _ = roc_curve(y[ite], p)
        ti = np.interp(grille, fpr, tpr)
        ti[0] = 0.0
        tprs.append(ti)
        aucs.append(float(_auc(fpr, tpr)))
    tprs = np.asarray(tprs)
    return grille, tprs.mean(0), tprs.std(0), float(np.mean(aucs)), float(np.std(aucs))


def fig_roc(avec: dict, sans: dict, sortie: Path) -> None:
    """Deux panneaux (combiné | habitat) : ROC moyenne avec vs sans élévation."""
    m = _module_modele()
    from config import load_config
    cfg = load_config(str(RACINE / "config.yaml"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    for ax, cle, titre in [(axes[0], "modele_combine", "Modèle combiné (habitat + détection)"),
                           (axes[1], "modele_habitat", "Modèle habitat (H1–H4)")]:
        for src, coul, lab in [(avec, C_AVEC, "Avec élévation"), (sans, C_SANS, "Sans élévation")]:
            g, tpr, std, a, asd = _roc_moyenne(m, cfg, src[cle]["features"],
                                               src[cle]["best_params"])
            ax.plot(g, tpr, color=coul, lw=2.0, label=f"{lab} — AUC {a:.3f} ± {asd:.3f}")
            ax.fill_between(g, np.clip(tpr - std, 0, 1), np.clip(tpr + std, 0, 1),
                            color=coul, alpha=0.14)
            log.info("    %s · %s : AUC %.3f ± %.3f", titre, lab, a, asd)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#9aa4ac")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.005)
        ax.set_aspect("equal")
        ax.set_title(titre, fontsize=10.5)
        ax.set_xlabel("Taux de faux positifs (1 − spécificité)")
        ax.legend(loc="lower right", frameon=False, fontsize=8.8)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=0.6)
        _sans_cadre(ax)
    axes[0].set_ylabel("Taux de vrais positifs (sensibilité)")
    fig.suptitle("Courbes ROC — validation croisée spatiale (moyenne ± σ sur 5 folds)",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(sortie, bbox_inches="tight")
    plt.close(fig)
    log.info("  écrit %s", sortie.name)


# ── B1 / B2 — Diagrammes de pipeline (matplotlib pur) ─────────────────────────
# Palette reprise du diagramme Mermaid du README (cohérence visuelle).
PAL = {
    "src":   ("#EEEDFE", "#534AB7", "#26215C"),
    "data":  ("#E1F5EE", "#0F6E56", "#04342C"),
    "model": ("#E6F1FB", "#185FA5", "#042C53"),
    "pred":  ("#FAEEDA", "#854F0B", "#412402"),
    "rap":   ("#FAECE7", "#993C1D", "#4A1B0C"),
}


def _boite(ax, x, y, w, h, titre, sous="", pal="data", ftitre=10, fsous=8.5):
    from matplotlib.patches import FancyBboxPatch
    fill, stroke, ink = PAL[pal]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                linewidth=1.6, edgecolor=stroke, facecolor=fill))
    cy = y + h / 2
    if sous:
        ax.text(x + w / 2, cy + h * 0.16, titre, ha="center", va="center",
                fontsize=ftitre, weight="bold", color=ink)
        ax.text(x + w / 2, cy - h * 0.22, sous, ha="center", va="center",
                fontsize=fsous, color=ink, wrap=True)
    else:
        ax.text(x + w / 2, cy, titre, ha="center", va="center", fontsize=ftitre,
                weight="bold", color=ink)
    return (x, y, w, h)


def _fleche(ax, p1, p2, style="-", coul="#5f5f5f", rad=0.0, txt="", lw=1.6):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15, linewidth=lw,
                                 color=coul, linestyle=style,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if txt:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 2.4 + rad * 18, txt, ha="center", va="bottom", fontsize=7.6,
                color=coul, style="italic")


def fig_pipeline(sortie: Path) -> None:
    """B1 — flux de traitement 01→05/08 (sources → prédicteurs/eBird → RF → cartes → figures)."""
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    src = _boite(ax, 2, 24, 18, 54,
                 "Sources de données",
                 "LiDAR — MHC · MNT\nÉcoforestière MFFP\nLandsat C2 L2 · STAC\n"
                 "AQréseau+ (routes)\nMilieux humides pot.\neBird EBD (checklists)",
                 "src", ftitre=10.5, fsous=8.8)
    pred = _boite(ax, 27, 55, 21, 21, "01_predictors.py",
                  "Stack 10 bandes\n5 m · EPSG:32198", "data")
    ebd = _boite(ax, 27, 24, 21, 21, "02_ebird.py",
                 "Table modèle\nzero-fill + covariables\n(buffer 30 m)", "data")
    mod = _boite(ax, 54, 39.5, 19, 21, "03_model.py",
                 "Random Forest\n2 modèles · CV spatiale\n(blocs 10 km)", "model")
    prd = _boite(ax, 78, 55, 20, 21, "04_predict.py",
                 "Cartes proba · incertitude\nMESS · hotspots", "pred")
    figb = _boite(ax, 78, 24, 20, 21, "05 + 08 · figures",
                  "Rapport &\nprésentation orale", "rap")
    # flux principal
    _fleche(ax, (src[0] + src[2], 62), (pred[0], 64))
    _fleche(ax, (src[0] + src[2], 40), (ebd[0], 36))
    _fleche(ax, (pred[0] + pred[2] / 2, pred[1]), (ebd[0] + ebd[2] / 2, ebd[1] + ebd[3]),
            txt="extraction covariables")
    _fleche(ax, (pred[0] + pred[2], 62), (mod[0], 54), rad=-0.15)
    _fleche(ax, (ebd[0] + ebd[2], 36), (mod[0], 46), rad=0.15)
    _fleche(ax, (mod[0] + mod[2], 52), (prd[0], 62), rad=-0.12)
    _fleche(ax, (pred[0] + pred[2], 66), (prd[0], 68), style="--", coul="#9a7b3a",
            txt="application fenêtrée")
    _fleche(ax, (prd[0] + prd[2] / 2, prd[1]), (figb[0] + figb[2] / 2, figb[1] + figb[3]))
    _fleche(ax, (mod[0] + mod[2] / 2, mod[1]), (figb[0] + 2, figb[1] + figb[3] / 2), rad=0.2)
    ax.set_title("Pipeline de traitement — SDM Engoulevent bois-pourri (5 m, EPSG:32198)",
                 fontsize=13, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(sortie, bbox_inches="tight")
    plt.close(fig)
    log.info("  écrit %s", sortie.name)


def fig_sources_variables(sortie: Path) -> None:
    """B2 — sources → 14 variables (10 habitat + 4 détection) → 2 modèles."""
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    sources = ["LiDAR (MRNF)", "Écoforestière (MFFP)", "Landsat C2 · STAC",
               "AQréseau+ (routes)", "Milieux humides pot.", "eBird — métadonnées"]
    ys = np.linspace(82, 12, len(sources))
    for s, y in zip(sources, ys):
        _boite(ax, 1.5, y - 4, 20, 8, s, "", "src", ftitre=8.8)
    habitat = ["Hauteur canopée (MHC)", "TWI", "Densité de lisière", "Prop. feuillu/mélangé",
               "Densité de routes", "LST estivale", "Classe d'âge", "Densité peuplement",
               "Distance milieu humide", "Élévation"]
    detection = ["Minutes après coucher", "Phase lunaire", "log(durée)", "Jour julien"]
    _boite(ax, 33, 46, 30, 46, "10 variables d'habitat",
           "\n".join(habitat) + "\n(cartographiées · 5 m)", "data", ftitre=10.5, fsous=8.2)
    _boite(ax, 33, 8, 30, 30, "4 variables de détection",
           "\n".join(detection) + "\n(RF seulement · non cartographiées)", "data",
           ftitre=10.5, fsous=8.2)
    _boite(ax, 73, 56, 25, 22, "Modèle combiné",
           "14 variables\n→ carte de probabilité\n(détection standardisée)", "model")
    _boite(ax, 73, 20, 25, 22, "Modèle habitat",
           "10 variables\n→ importance & PDP\ninterprétation H1–H4", "model")
    # sources → groupes de variables (faisceau simplifié)
    for y in ys[:5]:
        _fleche(ax, (21.5, y), (33, 69), coul="#b3adde", lw=1.1, rad=0.02)
    _fleche(ax, (21.5, ys[5]), (33, 23), coul="#b3adde", lw=1.1)
    # variables → modèles
    _fleche(ax, (63, 69), (73, 66), rad=-0.05)                # habitat → combiné
    _fleche(ax, (63, 23), (73, 62), rad=0.12)                 # détection → combiné
    _fleche(ax, (63, 60), (73, 31), rad=0.05)                 # habitat → habitat
    ax.set_title("Des sources aux 14 variables et aux deux modèles Random Forest",
                 fontsize=13, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(sortie, bbox_inches="tight")
    plt.close(fig)
    log.info("  écrit %s", sortie.name)


def main() -> None:
    import utils
    utils.setup_logging("08_figures_rapport", log_dir="outputs/comparaison/logs")
    _style()
    comp = RACINE / "outputs/comparaison/figures"
    comp.mkdir(parents=True, exist_ok=True)
    commun = RACINE / "outputs/commun/figures"
    commun.mkdir(parents=True, exist_ok=True)

    avec = _lire("outputs/avec_elevation/logs/03_model_rapport.json")
    sans = _lire("outputs/sans_elevation/logs/03_model_rapport.json")
    pred_a = _lire("outputs/avec_elevation/logs/04_predict_rapport.json")
    pred_s = _lire("outputs/sans_elevation/logs/04_predict_rapport.json")
    ebird = _lire("outputs/commun/logs/02_ebird_rapport.json")

    with utils.log_step("A1 — comparaison de performance", log):
        fig_performances(avec, sans, comp / "comparaison_performances.png")
    with utils.log_step("A2 — dominance de la détection", log):
        fig_importance_combine(avec, comp / "importance_combine.png")
    with utils.log_step("A3 — tableau de synthèse", log):
        fig_tableau_metriques(avec, sans, pred_a, pred_s, ebird, comp / "tableau_metriques.png")
    with utils.log_step("B4 — entonnoir eBird", log):
        fig_entonnoir(ebird, commun / "entonnoir_ebird.png")
    with utils.log_step("A4 — courbes ROC par fold (re-run CV)", log):
        fig_roc(avec, sans, comp / "roc_folds.png")
    with utils.log_step("B1 — diagramme de pipeline", log):
        fig_pipeline(comp / "pipeline.png")
    with utils.log_step("B2 — sources → variables → modèles", log):
        fig_sources_variables(comp / "sources_variables.png")


if __name__ == "__main__":
    main()
