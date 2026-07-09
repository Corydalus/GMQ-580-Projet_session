"""03 — Random Forest : tuning, validation croisée spatiale, importance, PDP.

Entraîne un `RandomForestClassifier` présence-absence sur `table_modele.parquet` (J4).
Validation croisée **spatiale** par blocs de `bloc_cv_km` km (StratifiedGroupKFold par
défaut — robuste au déséquilibre — ou GroupKFold), `RandomizedSearchCV` (scoring AUC-ROC),
métriques AUC-ROC + TSS par fold et importance par permutation. Deux modèles entraînés :
- **combiné** (14 var, approche Johnston) → carte J6, prédite à détectabilité standardisée ;
- **habitat seul** (10 var) → importance + PDP propres pour interpréter les hypothèses H1–H4,
  la détection dominant sinon la discrimination (engoulevent nocturne). Multi-cœurs sklearn (pas Dask).

Sorties : outputs/models/rf.joblib (modèle de prédiction + `detection_standard`),
outputs/models/rf_habitat.joblib, outputs/models/pdp_habitat_rapport.json,
outputs/logs/03_model_rapport.json.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

import utils
from config import Config, load_config_from_cli

log = logging.getLogger("03_model")

# 10 variables d'habitat (= bandes du stack, cartographiées en J6) + 4 variables de détection.
FEATURES_HABITAT = [
    "MHC_hauteur_canopee", "TWI", "densite_lisiere", "prop_feuillu_melange", "densite_routes",
    "LST_estivale", "classe_age", "densite_peuplement", "distance_milieu_humide", "elevation",
]
FEATURES_DETECTION = ["minutes_apres_coucher", "phase_lune", "log_duree", "jour_julien"]
FEATURES = FEATURES_HABITAT + FEATURES_DETECTION
CIBLE = "presence"


# ── Blocs spatiaux & métriques ───────────────────────────────────────────────

def assigner_blocs(x: np.ndarray, y: np.ndarray, taille_m: float) -> np.ndarray:
    """Identifiant entier du bloc spatial de `taille_m` (grille régulière) de chaque point."""
    x, y = np.asarray(x), np.asarray(y)
    bx = np.floor((x - x.min()) / taille_m).astype(np.int64)
    by = np.floor((y - y.min()) / taille_m).astype(np.int64)
    return bx * (by.max() + 1) + by


def tss_optimal(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """TSS maximal (sensibilité + spécificité − 1) et son seuil, via la courbe ROC."""
    from sklearn.metrics import roc_curve
    fpr, tpr, seuils = roc_curve(y_true, y_score)
    tss = tpr - fpr
    i = int(np.argmax(tss))
    return float(tss[i]), float(seuils[i])


def make_cv(cfg: Config):
    """Objet de CV spatiale selon `modele.cv_stratifie` (Stratified/GroupKFold)."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    n = cfg.modele.n_folds
    if cfg.modele.cv_stratifie:
        return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=cfg.modele.random_state)
    return GroupKFold(n_splits=n)


# ── Modèle & tuning ──────────────────────────────────────────────────────────

def _rf(cfg: Config, n_jobs: int = 1, **params):
    """RandomForestClassifier présence-absence (class_weight balanced, seed fixe)."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(class_weight="balanced", random_state=cfg.modele.random_state,
                                  n_jobs=n_jobs, **params)


def chercher_hyperparams(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv, cfg: Config):
    """RandomizedSearchCV (AUC-ROC) sur la CV spatiale. Retourne (best_params, best_score)."""
    from scipy.stats import randint
    from sklearn.model_selection import RandomizedSearchCV
    espace = {
        "n_estimators": randint(300, 1200),
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "min_samples_leaf": randint(1, 11),
        "max_depth": [None, 10, 20, 30, 40],
    }
    rech = RandomizedSearchCV(
        _rf(cfg, n_jobs=1), espace, n_iter=cfg.modele.n_iter_recherche, scoring="roc_auc",
        cv=cv, n_jobs=-1, random_state=cfg.modele.random_state, refit=False, error_score="raise",
    )
    rech.fit(X, y, groups=groups)
    return rech.best_params_, float(rech.best_score_)


def evaluer_spatial(params: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    cv, cfg: Config) -> dict:
    """AUC-ROC + TSS par fold de la CV spatiale (fit sur les blocs d'entraînement)."""
    from sklearn.metrics import roc_auc_score
    aucs, tss_list, seuils = [], [], []
    for k, (itr, ite) in enumerate(cv.split(X, y, groups), 1):
        mod = _rf(cfg, n_jobs=-1, **params).fit(X[itr], y[itr])
        p = mod.predict_proba(X[ite])[:, 1]
        auc = float(roc_auc_score(y[ite], p))
        tss, seuil = tss_optimal(y[ite], p)
        aucs.append(auc)
        tss_list.append(tss)
        seuils.append(seuil)
        log.info("  fold %d/%d : AUC=%.3f · TSS=%.3f (seuil=%.3f) · n_test=%d (%d prés.)",
                 k, cfg.modele.n_folds, auc, tss, seuil, len(ite), int(y[ite].sum()))
    arr = lambda v: [round(x, 4) for x in v]  # noqa: E731
    return dict(auc_folds=arr(aucs), auc_moy=round(float(np.mean(aucs)), 4),
                auc_std=round(float(np.std(aucs)), 4), tss_folds=arr(tss_list),
                tss_moy=round(float(np.mean(tss_list)), 4),
                tss_std=round(float(np.std(tss_list)), 4), seuils=arr(seuils))


# ── Importance & dépendance partielle ────────────────────────────────────────

def importance_permutation(model, X: np.ndarray, y: np.ndarray, cfg: Config,
                           features: list[str]) -> list[dict]:
    """Importance par permutation (AUC-ROC), triée décroissante, sur les variables de `features`."""
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10,
                               random_state=cfg.modele.random_state, n_jobs=-1)
    imp = [dict(variable=features[i], importance=round(float(r.importances_mean[i]), 5),
                ecart_type=round(float(r.importances_std[i]), 5)) for i in range(len(features))]
    return sorted(imp, key=lambda d: d["importance"], reverse=True)


def pdp_habitat(model, X: np.ndarray, features: list[str],
                features_habitat: list[str] | None = None) -> dict:
    """Dépendance partielle (grille + valeurs) des variables d'habitat de `features` (figures J7)."""
    from sklearn.inspection import partial_dependence
    features_habitat = features_habitat or FEATURES_HABITAT
    out = {}
    for nom in features_habitat:
        if nom not in features:
            continue
        j = features.index(nom)
        pd = partial_dependence(model, X, [j], grid_resolution=40, kind="average")
        out[nom] = dict(grille=[round(v, 4) for v in pd["grid_values"][0].tolist()],
                        pd=[round(v, 5) for v in pd["average"][0].tolist()])
    return out


# ── SHAP & diagnostic spatial de l'élévation ─────────────────────────────────

def echantillon_shap(y: np.ndarray, taille: int, random_state: int) -> np.ndarray:
    """Indices d'un échantillon SHAP : toutes les présences + des absences tirées (0 = tout)."""
    n = len(y)
    if not taille or taille >= n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    pres = np.where(y == 1)[0]
    absc = np.where(y == 0)[0]
    n_abs = min(max(taille - len(pres), 0), len(absc))
    return np.sort(np.concatenate([pres, rng.choice(absc, size=n_abs, replace=False)]))


def valeurs_shap(model, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Valeurs SHAP de la classe présence (TreeExplainer), robuste aux versions de shap."""
    import shap
    exp = shap.TreeExplainer(model)(X, check_additivity=False)
    vals, base = exp.values, np.asarray(exp.base_values)
    if vals.ndim == 3:  # (n, features, classes) → classe présence
        return vals[:, :, 1], float(base[..., 1].mean())
    return vals, float(base.mean())


def shap_habitat(model, X: np.ndarray, y: np.ndarray, cfg: Config,
                 features_habitat: list[str] | None = None) -> dict:
    """Valeurs SHAP (classe présence) du modèle habitat sur un échantillon + importance |SHAP| moyenne."""
    features_habitat = features_habitat or FEATURES_HABITAT
    idx = echantillon_shap(y, cfg.modele.shap_echantillon, cfg.modele.random_state)
    vals, base = valeurs_shap(model, X[idx])
    imp = sorted(
        [dict(variable=f, importance_shap=round(float(np.abs(vals[:, j]).mean()), 5))
         for j, f in enumerate(features_habitat)],
        key=lambda d: d["importance_shap"], reverse=True)
    return dict(values=vals, data=X[idx], presence=y[idx], base_value=base,
                mean_abs=imp, n_echantillon=int(len(idx)))


def _r2_lineaire(cible: np.ndarray, *regresseurs: np.ndarray) -> float:
    """R² d'une régression linéaire (moindres carrés) de `cible` sur les régresseurs + constante."""
    a = np.column_stack([np.ones_like(cible), *regresseurs])
    beta, *_ = np.linalg.lstsq(a, cible, rcond=None)
    ss_res = float(np.sum((cible - a @ beta) ** 2))
    ss_tot = float(np.sum((cible - cible.mean()) ** 2))
    return round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0


def diagnostic_elevation(X: np.ndarray, y: np.ndarray, groups: np.ndarray, df: pl.DataFrame,
                         cv, cfg: Config, best_params: dict, importance_habitat: list[dict],
                         features: list[str] | None = None,
                         features_habitat: list[str] | None = None) -> dict:
    """Teste si l'élévation agit comme proxy spatial.

    (1) structure spatiale de l'élévation (R² élévation~x,y ; corrélations) ;
    (2) ajout des coordonnées x,y au modèle habitat (mêmes hyperparams) → effet sur
    l'importance/rang de l'élévation et sur l'AUC spatiale. Une chute marquée de
    l'importance de l'élévation quand x,y sont présents signale un proxy spatial.
    """
    features = features or FEATURES
    features_habitat = features_habitat or FEATURES_HABITAT
    x, yc = df["x"].to_numpy(), df["y"].to_numpy()
    elev = X[:, features.index("elevation")]
    rang_sans = {d["variable"]: i + 1 for i, d in enumerate(importance_habitat)}
    imp_sans = {d["variable"]: d["importance"] for d in importance_habitat}

    xh = X[:, [features.index(f) for f in features_habitat]]
    xhxy = np.column_stack([xh, x, yc])
    feats_xy = features_habitat + ["x", "y"]
    mod_xy = _rf(cfg, n_jobs=-1, **best_params).fit(xhxy, y)
    imp_xy = importance_permutation(mod_xy, xhxy, y, cfg, feats_xy)
    rang_avec = {d["variable"]: i + 1 for i, d in enumerate(imp_xy)}
    imp_avec = {d["variable"]: d["importance"] for d in imp_xy}

    auc_sans = evaluer_spatial(best_params, xh, y, groups, cv, cfg)["auc_moy"]
    auc_avec = evaluer_spatial(best_params, xhxy, y, groups, cv, cfg)["auc_moy"]
    return dict(
        r2_elevation_xy=_r2_lineaire(elev, x, yc),
        corr_elevation_x=round(float(np.corrcoef(elev, x)[0, 1]), 4),
        corr_elevation_y=round(float(np.corrcoef(elev, yc)[0, 1]), 4),
        elevation_sans_coords=dict(importance=imp_sans["elevation"], rang=rang_sans["elevation"]),
        elevation_avec_coords=dict(importance=imp_avec["elevation"], rang=rang_avec["elevation"]),
        coords=dict(x=dict(importance=imp_avec["x"], rang=rang_avec["x"]),
                    y=dict(importance=imp_avec["y"], rang=rang_avec["y"])),
        auc_habitat=auc_sans, auc_habitat_xy=auc_avec,
        importance_avec_coords=imp_xy)


# ── Orchestration ────────────────────────────────────────────────────────────

def features_effectifs(cfg: Config) -> tuple[list[str], list[str]]:
    """(features, features_habitat) après exclusion éventuelle (cfg.variables.exclure).

    Permet d'explorer des jeux de variables réduits (p. ex. retirer l'élévation) via config,
    sans toucher au code. `exclure=[]` (défaut) → listes complètes, comportement inchangé.
    """
    exclure = set(cfg.variables.exclure)
    fh = [f for f in FEATURES_HABITAT if f not in exclure]
    return fh + FEATURES_DETECTION, fh


def charger_donnees(cfg: Config, features: list[str] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    """Charge la table modèle → (X, y, groups blocs, df). Vérifie l'absence de NaN."""
    features = features or FEATURES
    chemin = Path(cfg.chemins.processed) / "table_modele.parquet"
    if not chemin.exists():
        raise FileNotFoundError(f"Table modèle introuvable ({chemin}) — exécuter 02_ebird.py (J4).")
    df = pl.read_parquet(chemin)
    X = df.select(features).to_numpy()
    y = df[CIBLE].to_numpy().astype(int)
    if not np.isfinite(X).all():
        raise ValueError("X contient des valeurs non finies (le RF n'accepte pas les NaN).")
    groups = assigner_blocs(df["x"].to_numpy(), df["y"].to_numpy(), cfg.modele.bloc_cv_km * 1000)
    log.info("  %d checklists · %d présences · %d variables · %d blocs de %g km",
             len(y), int(y.sum()), len(features), len(np.unique(groups)), cfg.modele.bloc_cv_km)
    return X, y, groups, df


def entrainer(nom: str, features: list[str], X_full: np.ndarray, y: np.ndarray,
              groups: np.ndarray, cv, cfg: Config, avec_pdp: bool,
              features_full: list[str] | None = None,
              features_habitat: list[str] | None = None) -> tuple[dict, object, dict]:
    """Tuning + CV spatiale + fit final + importance (+ PDP) pour un jeu de variables.

    Retourne (métriques+importance, modèle ajusté, PDP habitat). `X_full` est en ordre
    `features_full` (défaut `FEATURES`) ; on en extrait les colonnes de `features`.
    """
    features_full = features_full or FEATURES
    features_habitat = features_habitat or FEATURES_HABITAT
    X = X_full[:, [features_full.index(f) for f in features]]
    with utils.log_step(f"[{nom}] Recherche d'hyperparamètres (RandomizedSearchCV)", log):
        best_params, best_score = chercher_hyperparams(X, y, groups, cv, cfg)
        log.info("  [%s] meilleurs params : %s (AUC CV=%.3f)", nom, best_params, best_score)
    with utils.log_step(f"[{nom}] Validation croisée spatiale (AUC + TSS)", log):
        metriques = evaluer_spatial(best_params, X, y, groups, cv, cfg)
        log.info("  [%s] AUC=%.3f±%.3f · TSS=%.3f±%.3f", nom, metriques["auc_moy"],
                 metriques["auc_std"], metriques["tss_moy"], metriques["tss_std"])
    with utils.log_step(f"[{nom}] Ajustement final + importance par permutation", log):
        modele = _rf(cfg, n_jobs=-1, **best_params).fit(X, y)
        importance = importance_permutation(modele, X, y, cfg, features)
        log.info("  [%s] top 3 : %s", nom, ", ".join(f"{d['variable']}={d['importance']:.3f}"
                                                      for d in importance[:3]))
    pdp = {}
    if avec_pdp:
        with utils.log_step(f"[{nom}] Dépendance partielle (variables d'habitat)", log):
            pdp = pdp_habitat(modele, X, features, features_habitat)
    resume = dict(features=features, best_params=best_params, best_score_cv=round(best_score, 4),
                  **metriques, importance=importance)
    return resume, modele, pdp


def detection_standard(df: pl.DataFrame) -> dict:
    """Valeurs de détection « au crépuscule » (médiane des présences) pour la prédiction J6."""
    pres = df.filter(pl.col(CIBLE) == 1)
    return {v: round(float(pres[v].median()), 4) for v in FEATURES_DETECTION}


def main() -> None:
    import joblib

    cfg = load_config_from_cli()
    utils.setup_logging("03_model", log_dir=f"{cfg.chemins.outputs}/logs")
    rapport: dict = {}

    features, features_habitat = features_effectifs(cfg)
    if cfg.variables.exclure:
        log.info("  variables exclues (config) : %s → %d var (%d habitat + %d détection)",
                 cfg.variables.exclure, len(features), len(features_habitat), len(FEATURES_DETECTION))
    rapport["variables_exclues"] = list(cfg.variables.exclure)

    with utils.log_step("Chargement de la table modèle", log):
        X, y, groups, df = charger_donnees(cfg, features)
        rapport.update(n_checklists=len(y), n_presences=int(y.sum()), n_absences=int((y == 0).sum()),
                       n_blocs=int(len(np.unique(groups))), cv_stratifie=cfg.modele.cv_stratifie,
                       bloc_cv_km=cfg.modele.bloc_cv_km)
    cv = make_cv(cfg)

    # Modèle de PRÉDICTION (habitat + détection, approche Johnston : carte J6 à détectabilité fixée).
    combine, modele_combine, _ = entrainer("combiné", features, X, y, groups, cv, cfg,
                                           avec_pdp=False, features_full=features,
                                           features_habitat=features_habitat)
    # Modèle d'INTERPRÉTATION habitat : importance + PDP propres des hypothèses H1–H4.
    habitat, modele_habitat, pdp = entrainer("habitat", features_habitat, X, y, groups, cv, cfg,
                                             avec_pdp=True, features_full=features,
                                             features_habitat=features_habitat)
    rapport["modele_combine"] = combine
    rapport["modele_habitat"] = habitat
    det_std = detection_standard(df)
    rapport["detection_standard"] = det_std

    models_dir = Path(cfg.chemins.outputs) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Interprétation SHAP du modèle habitat (valeurs de Shapley, classe présence).
    x_hab = X[:, [features.index(f) for f in features_habitat]]
    with utils.log_step("Valeurs SHAP (modèle habitat)", log):
        sh = shap_habitat(modele_habitat, x_hab, y, cfg, features_habitat)
        np.savez_compressed(models_dir / "shap_habitat.npz", values=sh["values"], data=sh["data"],
                            presence=sh["presence"], base_value=sh["base_value"],
                            features=np.array(features_habitat))
        rapport["shap_habitat"] = dict(mean_abs=sh["mean_abs"], n_echantillon=sh["n_echantillon"],
                                       base_value=round(sh["base_value"], 5),
                                       npz=str(models_dir / "shap_habitat.npz"))
        log.info("  SHAP top 3 (|SHAP| moy) : %s", ", ".join(
            f"{d['variable']}={d['importance_shap']:.3f}" for d in sh["mean_abs"][:3]))

    # Diagnostic : l'élévation est-elle un proxy spatial ? (ajout des coordonnées x,y)
    if cfg.modele.diagnostic_spatial and "elevation" in features_habitat:
        with utils.log_step("Diagnostic spatial de l'élévation (ajout x,y)", log):
            diag = diagnostic_elevation(X, y, groups, df, cv, cfg, habitat["best_params"],
                                        habitat["importance"], features, features_habitat)
            rapport["diagnostic_elevation"] = diag
            log.info("  élévation : imp %.3f (rang %d) → %.3f (rang %d) avec x,y ; "
                     "R²(élév~x,y)=%.2f ; AUC habitat %.3f → %.3f (+x,y)",
                     diag["elevation_sans_coords"]["importance"],
                     diag["elevation_sans_coords"]["rang"],
                     diag["elevation_avec_coords"]["importance"],
                     diag["elevation_avec_coords"]["rang"], diag["r2_elevation_xy"],
                     diag["auc_habitat"], diag["auc_habitat_xy"])

    with utils.log_step("Export des modèles et des PDP", log):
        # rf.joblib = modèle de prédiction (J6) ; détection à fixer à `detection_standard`.
        joblib.dump(dict(modele=modele_combine, features=features,
                         features_habitat=features_habitat, features_detection=FEATURES_DETECTION,
                         best_params=combine["best_params"], detection_standard=det_std,
                         seuil_tss_median=float(np.median(combine["seuils"]))),
                    models_dir / "rf.joblib")
        joblib.dump(dict(modele=modele_habitat, features=features_habitat,
                         best_params=habitat["best_params"],
                         seuil_tss_median=float(np.median(habitat["seuils"]))),
                    models_dir / "rf_habitat.joblib")
        utils.ecrire_rapport_json("pdp_habitat", pdp, log_dir=str(models_dir))
        rapport.update(rf_joblib=str(models_dir / "rf.joblib"),
                       rf_habitat_joblib=str(models_dir / "rf_habitat.joblib"),
                       pdp=str(models_dir / "pdp_habitat_rapport.json"))

    utils.ecrire_rapport_json("03_model", rapport, log_dir=f"{cfg.chemins.outputs}/logs")


if __name__ == "__main__":
    main()
