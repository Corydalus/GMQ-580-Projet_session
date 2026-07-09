"""06 — Exploration : impact de l'élévation sur le modèle habitat (retrait / résidualisation).

Le diagnostic J5 a montré que l'élévation est spatialement confondue (R²(élévation~x,y)=0.36 ;
importance par permutation 0.115 → 0.030, soit −74 %, dès qu'on ajoute x,y). On évalue ici le
potentiel de présence basé **majoritairement sur l'habitat** en comparant, sur la **même CV
spatiale**, trois variantes du modèle habitat :

- **baseline** — 10 variables d'habitat (dont l'élévation brute) ;
- **sans élévation** — 9 variables (élévation retirée) ;
- **élévation résidualisée** — l'élévation est remplacée par son **anomalie locale**
  (élévation − tendance planaire x,y), qui retire la composante spatiale (~36 %) tout en
  conservant le relief local réel.

Chaque variante est ré-ajustée (RandomizedSearchCV sur la CV spatiale) puis comparée en
AUC-ROC + TSS, importance par permutation et |SHAP| moyen. C'est une **exploration** (branche
`explore/elevation-habitat`), pas le pipeline de production.

Sorties : outputs/figures/exploration_elevation.png, outputs/logs/06_exploration_rapport.json
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import utils  # noqa: E402
from config import load_config_from_cli  # noqa: E402

log = logging.getLogger("06_exploration")

ETIQUETTES = {
    "MHC_hauteur_canopee": "Hauteur canopée", "TWI": "TWI", "densite_lisiere": "Densité lisière",
    "prop_feuillu_melange": "Prop. feuillu/mél.", "densite_routes": "Densité routes",
    "LST_estivale": "LST estivale", "classe_age": "Classe d'âge",
    "densite_peuplement": "Densité peupl.", "distance_milieu_humide": "Dist. milieu humide",
    "elevation": "Élévation", "elevation_residuelle": "Élévation (anomalie locale)",
}


def charger_module_modele():
    """Charge code/03_model.py (préfixe numérique → importlib) pour réutiliser ses fonctions."""
    chemin = Path(__file__).resolve().parent / "03_model.py"
    spec = importlib.util.spec_from_file_location("model03", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def residualiser(colonne: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résidu d'une régression linéaire de `colonne` sur (x, y) : anomalie locale sans tendance spatiale."""
    a = np.column_stack([np.ones_like(colonne), x, y])
    beta, *_ = np.linalg.lstsq(a, colonne, rcond=None)
    return colonne - a @ beta


def variantes_habitat(m, x_hab: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Trois matrices de variables habitat : baseline, sans élévation, élévation résidualisée."""
    feats = list(m.FEATURES_HABITAT)
    je = feats.index("elevation")
    feats_sans = [f for f in feats if f != "elevation"]
    feats_resid = [f if f != "elevation" else "elevation_residuelle" for f in feats]
    x_resid = x_hab.copy()
    x_resid[:, je] = residualiser(x_hab[:, je], x, y)
    return {
        "baseline": (x_hab, feats),
        "sans_elevation": (np.delete(x_hab, je, axis=1), feats_sans),
        "elevation_residualisee": (x_resid, feats_resid),
    }


def evaluer_variante(m, nom: str, X: np.ndarray, feats: list[str], y: np.ndarray,
                     groups: np.ndarray, cv, cfg) -> dict:
    """Tuning + CV spatiale + importance + |SHAP| moyen d'une variante habitat."""
    with utils.log_step(f"[{nom}] tuning + CV spatiale + importance", log):
        best_params, best_score = m.chercher_hyperparams(X, y, groups, cv, cfg)
        metriques = m.evaluer_spatial(best_params, X, y, groups, cv, cfg)
        modele = m._rf(cfg, n_jobs=-1, **best_params).fit(X, y)
        importance = m.importance_permutation(modele, X, y, cfg, feats)
        idx = m.echantillon_shap(y, cfg.modele.shap_echantillon, cfg.modele.random_state)
        vals, _ = m.valeurs_shap(modele, X[idx])
        shap_moy = {feats[j]: round(float(np.abs(vals[:, j]).mean()), 5) for j in range(len(feats))}
        log.info("  [%s] AUC=%.3f±%.3f · TSS=%.3f±%.3f · top: %s", nom, metriques["auc_moy"],
                 metriques["auc_std"], metriques["tss_moy"], metriques["tss_std"],
                 ", ".join(f"{d['variable']}={d['importance']:.3f}" for d in importance[:3]))
    return dict(features=feats, best_params=best_params, best_score_cv=round(best_score, 4),
                **metriques, importance=importance, shap_moy=shap_moy)


def fig_comparaison(resultats: dict, sortie: str) -> str:
    """AUC/TSS par variante (gauche) + importance par permutation redistribuée (droite)."""
    noms = list(resultats)
    couleurs = {"baseline": "#2c7fb8", "sans_elevation": "#d95f0e",
                "elevation_residualisee": "#41ab5d"}
    fig, (axm, axi) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [1, 1.4]})

    # (1) AUC-ROC & TSS ± écart-type
    largeur = 0.35
    xpos = np.arange(len(noms))
    for k, (cle, lab) in enumerate((("auc", "AUC-ROC"), ("tss", "TSS"))):
        moy = [resultats[n][f"{cle}_moy"] for n in noms]
        err = [resultats[n][f"{cle}_std"] for n in noms]
        barres = axm.bar(xpos + (k - 0.5) * largeur, moy, largeur, yerr=err, capsize=4,
                         label=lab, color="#3690c0" if k == 0 else "#d95f0e")
        for b, v in zip(barres, moy):
            axm.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                         va="bottom", fontsize=8)
    axm.set_xticks(xpos)
    axm.set_xticklabels([n.replace("_", "\n") for n in noms], fontsize=9)
    axm.set_ylim(0, 1.0)
    axm.set_ylabel("Score (CV spatiale 10 km)")
    axm.set_title("Discrimination par variante")
    axm.legend(fontsize=9)
    axm.grid(axis="y", alpha=0.3)

    # (2) importance par permutation — variables d'habitat hors élévation, par variante
    base_imp = {d["variable"]: d["importance"] for d in resultats["baseline"]["importance"]}
    vars_communes = [v for v in sorted(base_imp, key=base_imp.get, reverse=True)
                     if v not in ("elevation", "elevation_residuelle")]
    ypos = np.arange(len(vars_communes))
    h = 0.26
    for k, n in enumerate(noms):
        imp = {d["variable"]: d["importance"] for d in resultats[n]["importance"]}
        vals = [imp.get(v, 0.0) for v in vars_communes]
        axi.barh(ypos + (1 - k) * h, vals, h, label=n, color=couleurs[n])
    axi.set_yticks(ypos)
    axi.set_yticklabels([ETIQUETTES.get(v, v) for v in vars_communes], fontsize=8)
    axi.invert_yaxis()
    axi.set_xlabel("Importance par permutation (chute d'AUC-ROC)")
    axi.set_title("Importance des variables d'habitat (hors élévation)")
    axi.legend(fontsize=8)
    axi.grid(axis="x", alpha=0.3)

    fig.suptitle("Impact de l'élévation sur le modèle habitat — retrait / résidualisation",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return sortie


def main() -> None:
    cfg = load_config_from_cli()
    utils.setup_logging("06_exploration", log_dir=f"{cfg.chemins.outputs}/logs")
    m = charger_module_modele()

    with utils.log_step("Chargement de la table modèle", log):
        X, y, groups, df = m.charger_donnees(cfg)
        x_hab = X[:, [m.FEATURES.index(f) for f in m.FEATURES_HABITAT]]
        xc, yc = df["x"].to_numpy(), df["y"].to_numpy()
    cv = m.make_cv(cfg)

    resultats = {}
    for nom, (Xv, feats) in variantes_habitat(m, x_hab, xc, yc).items():
        resultats[nom] = evaluer_variante(m, nom, Xv, feats, y, groups, cv, cfg)

    fig_dir = Path(cfg.chemins.outputs) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sortie = fig_comparaison(resultats, str(fig_dir / "exploration_elevation.png"))
    log.info("  figure → %s", sortie)

    rapport = {nom: {k: r[k] for k in ("best_params", "best_score_cv", "auc_moy", "auc_std",
                                       "tss_moy", "tss_std", "importance", "shap_moy")}
               for nom, r in resultats.items()}
    rapport["figure"] = sortie
    utils.ecrire_rapport_json("06_exploration", rapport, log_dir=f"{cfg.chemins.outputs}/logs")


if __name__ == "__main__":
    main()
