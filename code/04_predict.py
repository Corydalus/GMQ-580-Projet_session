"""04 — Prédiction : carte proba (fenêtrée) + incertitude + MESS + hotspots.

Applique le modèle RF combiné (`rf.joblib`, J5) au stack 5 m, tuile par tuile, la
détection **fixée à `detection_standard`** (engoulevent au crépuscule). Produit :
- carte de probabilité de présence (moyenne des arbres),
- carte d'incertitude (variance inter-arbres),
- carte MESS (Elith et al. 2010) — zones d'extrapolation hors domaine d'entraînement,
- hotspots : cellules 1 km à haute probabilité et faible effort eBird (sous-échantillonnées).

Le seuil de binarisation est **recalculé à détection fixée** (l'échelle de proba diffère
de la CV, où la détection variait). Rasters COG DEFLATE 512 ; boucle inter-arbres threadée.

Sorties : outputs/maps/{proba,incertitude,mess}_5m.tif (COG),
outputs/tables/hotspots.{gpkg,csv}, outputs/logs/04_predict_rapport.json.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import polars as pl

import utils
from config import Config, load_config_from_cli

log = logging.getLogger("04_predict")

FEATURES_HABITAT = [
    "MHC_hauteur_canopee", "TWI", "densite_lisiere", "prop_feuillu_melange", "densite_routes",
    "LST_estivale", "classe_age", "densite_peuplement", "distance_milieu_humide", "elevation",
]
FEATURES_DETECTION = ["minutes_apres_coucher", "phase_lune", "log_duree", "jour_julien"]


# ── Prédiction fenêtrée ──────────────────────────────────────────────────────

def predire_moyenne_variance(rf, X: np.ndarray, parallel=None) -> tuple[np.ndarray, np.ndarray]:
    """Proba de présence (moyenne des arbres) et variance inter-arbres, sur X (n, 14).

    Boucle threadée sur les arbres (predict_proba Cython relâche le GIL). `parallel` =
    contexte joblib.Parallel réutilisé (sinon séquentiel).
    """
    n_trees = len(rf.estimators_)
    preds = np.empty((n_trees, X.shape[0]), dtype="float32")

    def _un(i, arbre):
        preds[i] = arbre.predict_proba(X)[:, 1]

    if parallel is None:
        for i, arbre in enumerate(rf.estimators_):
            _un(i, arbre)
    else:
        from joblib import delayed
        parallel(delayed(_un)(i, arbre) for i, arbre in enumerate(rf.estimators_))
    return preds.mean(axis=0), preds.var(axis=0)


def _vecteur_detection(det_std: dict) -> np.ndarray:
    """Les 4 constantes de détection, dans l'ordre FEATURES_DETECTION."""
    return np.array([det_std[f] for f in FEATURES_DETECTION], dtype="float32")


def predire_stack(cfg: Config, stack_path: str, rf, det_std: dict, ref_habitat: list,
                  sorties: dict, rapport: dict) -> None:
    """Prédit proba + incertitude + MESS sur tout le stack (fenêtré) → 3 COG."""
    import rasterio
    import rasterio.shutil
    from joblib import Parallel

    det_vec = _vecteur_detection(det_std)
    interim = Path(cfg.chemins.interim)
    tmp = {k: interim / f"_pred_{k}.tif" for k in sorties}
    tuile = 1024
    n_neg_mess = n_valide = 0

    with rasterio.open(stack_path) as src:
        noms = list(src.descriptions)
        if noms != FEATURES_HABITAT:
            raise ValueError(f"Ordre des bandes du stack {noms} ≠ {FEATURES_HABITAT}")
        profil = dict(driver="GTiff", width=src.width, height=src.height, count=1, dtype="float32",
                      crs=src.crs, transform=src.transform, nodata=float("nan"), tiled=True,
                      blockxsize=512, blockysize=512, compress="DEFLATE", BIGTIFF="YES")
        dst = {k: rasterio.open(tmp[k], "w", **profil) for k in sorties}
        try:
            with Parallel(n_jobs=-1, prefer="threads") as parallel:
                fenetres = list(utils.iter_windows(src.width, src.height, tuile))
                for n, win in enumerate(fenetres, 1):
                    bandes = src.read(window=win).astype("float32")          # (10, h, w)
                    h, w = bandes.shape[1:]
                    valide = np.isfinite(bandes).all(axis=0)                  # nodata en union
                    proba = np.full((h, w), np.nan, "float32")
                    var = np.full((h, w), np.nan, "float32")
                    if valide.any():
                        Xh = bandes[:, valide].T                             # (n_valide, 10)
                        X = np.hstack([Xh, np.broadcast_to(det_vec, (Xh.shape[0], 4))])
                        p, v = predire_moyenne_variance(rf, X, parallel)
                        proba[valide] = p
                        var[valide] = v
                        n_valide += int(valide.sum())
                    mess = utils.mess(bandes, ref_habitat)                    # (h, w)
                    mess[~valide] = np.nan
                    n_neg_mess += int((mess < 0).sum())
                    dst["proba"].write(proba, 1, window=win)
                    dst["incertitude"].write(var, 1, window=win)
                    dst["mess"].write(mess, 1, window=win)
                    if n % 50 == 0 or n == len(fenetres):
                        log.info("  tuile %d/%d", n, len(fenetres))
        finally:
            for d in dst.values():
                d.close()

    for k, chemin in sorties.items():
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        rasterio.shutil.copy(tmp[k], chemin, driver="COG", compress="DEFLATE", blocksize=512,
                             BIGTIFF="IF_SAFER")
        tmp[k].unlink(missing_ok=True)
    rapport.update(n_pixels_valides=n_valide,
                   pct_mess_negatif=round(n_neg_mess / n_valide * 100, 2) if n_valide else None)


# ── Seuil à détection fixée ──────────────────────────────────────────────────

def seuil_tss_detection_fixee(rf, df: pl.DataFrame, det_std: dict) -> tuple[float, float]:
    """TSS-optimal et son seuil sur les checklists, proba prédite à `detection_standard`."""
    from sklearn.metrics import roc_curve
    Xh = df.select(FEATURES_HABITAT).to_numpy()
    X = np.hstack([Xh, np.broadcast_to(_vecteur_detection(det_std), (Xh.shape[0], 4))])
    p = rf.predict_proba(X)[:, 1]
    fpr, tpr, seuils = roc_curve(df["presence"].to_numpy(), p)
    i = int(np.argmax(tpr - fpr))
    return float((tpr - fpr)[i]), float(seuils[i])


# ── Hotspots (haute proba × faible effort eBird) ─────────────────────────────

def hotspots(cfg: Config, proba_path: str, df: pl.DataFrame, seuil: float,
             n_hotspots: int = 5) -> tuple[object, dict]:
    """Cellules 1 km à proba ≥ seuil et effort eBird faible → n meilleurs (GeoDataFrame, stats)."""
    import geopandas as gpd
    import rasterio
    from rasterio.enums import Resampling
    from shapely.geometry import box

    pas = 1000.0
    with rasterio.open(proba_path) as ds:
        left, top = ds.transform.c, ds.transform.f
        res = ds.res[0]
        fac = int(round(pas / res))
        cols1 = math.ceil(ds.width / fac)
        rows1 = math.ceil(ds.height / fac)
        proba1 = ds.read(1, out_shape=(rows1, cols1), resampling=Resampling.average)
        crs = ds.crs

    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    c1 = np.floor((x - left) / pas).astype(int)
    r1 = np.floor((top - y) / pas).astype(int)
    effort = np.zeros((rows1, cols1), dtype="int32")
    dedans = (r1 >= 0) & (r1 < rows1) & (c1 >= 0) & (c1 < cols1)
    np.add.at(effort, (r1[dedans], c1[dedans]), 1)

    valide = np.isfinite(proba1)
    e_seuil = float(np.quantile(effort[valide], 0.25)) if valide.any() else 0.0
    cand = valide & (proba1 >= seuil) & (effort <= e_seuil)
    rr, cc = np.where(cand)
    ordre = np.argsort(proba1[rr, cc])[::-1][:n_hotspots]
    rr, cc = rr[ordre], cc[ordre]

    lignes, geoms = [], []
    for rang, (r, c) in enumerate(zip(rr, cc), 1):
        x0, y0 = left + c * pas, top - (r + 1) * pas
        lignes.append(dict(rang=rang, proba=round(float(proba1[r, c]), 4),
                           effort_ebird=int(effort[r, c]),
                           x_centre=round(x0 + pas / 2, 1), y_centre=round(y0 + pas / 2, 1)))
        geoms.append(box(x0, y0, x0 + pas, y0 + pas))
    gdf = gpd.GeoDataFrame(lignes, geometry=geoms, crs=crs)
    stats = dict(seuil_proba=round(seuil, 4), effort_seuil=e_seuil, n_hotspots=len(gdf))
    return gdf, stats


# ── Orchestration ────────────────────────────────────────────────────────────

def main() -> None:
    import joblib

    cfg = load_config_from_cli()
    utils.setup_logging("04_predict", log_dir=f"{cfg.chemins.outputs}/logs")
    rapport: dict = {}

    stack_path = f"{cfg.chemins.processed}/stack_5m.tif"
    rf_path = f"{cfg.chemins.outputs}/models/rf.joblib"
    table_path = f"{cfg.chemins.processed}/table_modele.parquet"
    for p in (stack_path, rf_path, table_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"Entrée manquante : {p} (exécuter J3/J4/J5).")

    paquet = joblib.load(rf_path)
    rf, det_std = paquet["modele"], paquet["detection_standard"]
    df = pl.read_parquet(table_path)
    ref_habitat = [df[c].to_numpy() for c in FEATURES_HABITAT]

    with utils.log_step("Seuil TSS à détection fixée", log):
        tss, seuil = seuil_tss_detection_fixee(rf, df, det_std)
        log.info("  TSS=%.3f · seuil=%.3f (détection = %s)", tss, seuil, det_std)
        rapport.update(tss_detection_fixee=round(tss, 4), seuil_detection_fixee=round(seuil, 4),
                       detection_standard=det_std)

    maps_dir = Path(cfg.chemins.outputs) / "maps"
    sorties = {k: str(maps_dir / f"{k}_5m.tif") for k in ("proba", "incertitude", "mess")}
    with utils.log_step("Prédiction fenêtrée (proba + incertitude + MESS)", log):
        predire_stack(cfg, stack_path, rf, det_std, ref_habitat, sorties, rapport)
        log.info("  %d px valides · %.1f%% MESS négatif (extrapolation)",
                 rapport["n_pixels_valides"], rapport["pct_mess_negatif"])
    rapport["cartes"] = sorties

    with utils.log_step("Hotspots (haute proba × faible effort)", log):
        gdf, stats = hotspots(cfg, sorties["proba"], df, seuil)
        tables_dir = Path(cfg.chemins.outputs) / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_file(tables_dir / "hotspots.gpkg", driver="GPKG")
        gdf.drop(columns="geometry").to_csv(tables_dir / "hotspots.csv", index=False)
        log.info("  %d hotspots (seuil proba=%.3f, effort ≤ %g)", stats["n_hotspots"],
                 stats["seuil_proba"], stats["effort_seuil"])
        rapport.update(stats, hotspots_gpkg=str(tables_dir / "hotspots.gpkg"))

    utils.ecrire_rapport_json("04_predict", rapport, log_dir=f"{cfg.chemins.outputs}/logs")


if __name__ == "__main__":
    main()
