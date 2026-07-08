"""01 — Prédicteurs : construit le stack raster 5 m (10 bandes) du SDM Engoulevent.

Aligne toutes les bandes sur la grille de référence (= grille du composite LST J2,
5 m EPSG:32198) : LiDAR (MHC, élévation, TWI), écoforestière MFFP (âge, densité,
feuillu/mélangé, lisière), densité de routes, distance aux milieux humides, et la
bande LST. Lecture fenêtrée uniquement (jamais la mosaïque complète en RAM).

Sortie : data/processed/stack_5m.tif (COG DEFLATE 512).

Construit par sous-étapes (voir CLAUDE.md §5, J3) :
  1. Fondations + LiDAR (MHC, élévation)  ← en cours
  2. TWI (WhiteboxTools)
  3. Écoforestière (âge, densité, feuillu, lisière)
  4. Routes + distance milieux humides
  5. Assemblage du stack 10 bandes
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import utils
from config import Config, load_config_from_cli

log = logging.getLogger("01_predictors")

# Ordre canonique des 10 bandes du stack (voir CLAUDE.md §2) : (fichier interim, description).
BANDES_STACK = [
    ("mhc_5m", "MHC_hauteur_canopee"),          # 1  H4
    ("twi_5m", "TWI"),                           # 2  H2
    ("lisiere_5m", "densite_lisiere"),           # 3  H1
    ("prop_feuillu_5m", "prop_feuillu_melange"), # 4  composition
    ("densite_routes_5m", "densite_routes"),     # 5  H3
    ("lst_estival_5m", "LST_estivale"),          # 6  limite thermique
    ("age_5m", "classe_age"),                    # 7  H4
    ("densite_5m", "densite_peuplement"),        # 8  H4
    ("distance_mh_5m", "distance_milieu_humide"),# 9  alimentation
    ("mnt_5m", "elevation"),                     # 10 limite altitudinale
]


def grille_reference(cfg: Config):
    """Grille de référence = composite LST J2 (5 m EPSG:32198). Retourne (chemin, (crs, tr, w, h))."""
    ref = f"{cfg.chemins.interim}/lst_estival_5m.tif"
    return ref, utils.lire_grille_reference(ref)


def agreger_lidar(cfg: Config, sous_dossier: str, motif: str, sortie: str) -> str:
    """Agrège les feuillets LiDAR `data/<sous_dossier>/<motif>*.tif` (1 m) sur la grille 5 m."""
    if Path(sortie).exists():
        log.info("%s déjà présent → réutilisation", sortie)
        return sortie
    _, (crs, transform, w, h) = grille_reference(cfg)
    data_root = Path(cfg.chemins.interim).parent
    tuiles = sorted((data_root / sous_dossier).glob(f"{motif}*.tif"))
    if not tuiles:
        raise FileNotFoundError(f"Aucun feuillet {motif}*.tif dans {data_root / sous_dossier}")
    log.info("%s : %d feuillets → agrégation moyenne à 5 m", sous_dossier, len(tuiles))
    arr = utils.agreger_tuiles_vers_grille(tuiles, crs, transform, w, h, resampling="average")
    utils.write_cog(sortie, arr, transform, crs=str(crs), nodata=float("nan"))
    log.info("  → %s (%.1f%% hors emprise LiDAR)", sortie, float(np.isnan(arr).mean() * 100))
    return sortie


def sous_etape_1_lidar(cfg: Config, rapport: dict) -> None:
    """MHC (var 1) et élévation/MNT (var 10) agrégés à 5 m sur la grille de référence."""
    with utils.log_step("MHC 1 m → 5 m", log):
        rapport["mhc_5m"] = agreger_lidar(cfg, "mhc", "MHC_",
                                          f"{cfg.chemins.interim}/mhc_5m.tif")
    with utils.log_step("Élévation (MNT) 1 m → 5 m", log):
        rapport["mnt_5m"] = agreger_lidar(cfg, "mnt", "MNT_",
                                          f"{cfg.chemins.interim}/mnt_5m.tif")


def sous_etape_2_twi(cfg: Config, rapport: dict) -> None:
    """TWI (var 2) via WhiteboxTools sur le MNT 5 m : breach → flow accum → slope → wetness."""
    import shutil

    import rasterio

    interim = Path(cfg.chemins.interim)
    crs = str(cfg.zone_etude.crs)
    sortie_twi = interim / "twi_5m.tif"
    if sortie_twi.exists():
        log.info("%s déjà présent → réutilisation", sortie_twi)
        rapport["twi_5m"] = str(sortie_twi)
        return
    with utils.log_step("TWI (WhiteboxTools)", log):
        tmp = interim / "_twi_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        # 1) MNT avec nodata numérique (-9999) — WhiteboxTools ne gère pas le NaN
        with rasterio.open(interim / "mnt_5m.tif") as ds:
            arr = ds.read(1)
            profil = ds.profile
        arr = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
        profil.update(driver="GTiff", nodata=-9999.0, compress="DEFLATE")
        with rasterio.open(tmp / "mnt.tif", "w", **profil) as ds:
            ds.write(arr, 1)
        # 2) chaîne hydrologique
        utils.wbt_run("BreachDepressionsLeastCost", dem=tmp / "mnt.tif",
                      output=tmp / "demf.tif", dist=cfg.lidar.twi_breach_dist_px)
        utils.wbt_run("D8FlowAccumulation", input=tmp / "demf.tif", output=tmp / "sca.tif",
                      out_type="specific contributing area")
        utils.wbt_run("Slope", dem=tmp / "demf.tif", output=tmp / "slope.tif", units="degrees")
        utils.wbt_run("WetnessIndex", sca=tmp / "sca.tif", slope=tmp / "slope.tif",
                      output=tmp / "twi.tif")
        # 3) nodata → NaN, écriture COG alignée
        with rasterio.open(tmp / "twi.tif") as ds:
            twi = ds.read(1, masked=True).filled(np.nan).astype("float32")
            transform = ds.transform
        sortie = interim / "twi_5m.tif"
        utils.write_cog(sortie, twi, transform, crs=crs, nodata=float("nan"))
        shutil.rmtree(tmp, ignore_errors=True)
        rapport["twi_5m"] = str(sortie)
        log.info("  → %s (%.1f%% valides)", sortie, float(np.isfinite(twi).mean() * 100))


def _ecrire_bande(sortie: str, arr: np.ndarray, valide: np.ndarray, transform, crs: str) -> str:
    """Écrit un COG float32, NaN hors `valide` (emprise de la source)."""
    band = np.where(valide, arr.astype("float32"), np.nan).astype("float32")
    utils.write_cog(sortie, band, transform, crs=crs, nodata=float("nan"))
    return sortie


def sous_etape_3_ecoforestiere(cfg: Config, rapport: dict) -> None:
    """Écoforestière MFFP → var 7 âge, 8 densité, 4 prop. feuillu/mélangé, 3 densité lisière."""
    import geopandas as gpd
    from scipy.ndimage import binary_dilation, uniform_filter

    import ecoforestiere as eco

    interim = cfg.chemins.interim
    noms = ("age_5m", "densite_5m", "prop_feuillu_5m", "lisiere_5m")
    sorties = {n: f"{interim}/{n}.tif" for n in noms}
    if all(Path(s).exists() for s in sorties.values()):
        log.info("écoforestière déjà rasterisée → réutilisation")
        rapport.update(sorties)
        return

    _, (_, transform, w, h) = grille_reference(cfg)
    crs = str(cfg.zone_etude.crs)
    res = cfg.zone_etude.resolution_m

    with utils.log_step("Chargement écoforestière (pee)", log):
        gdf = gpd.read_file(cfg.sources.ecoforestiere, layer=cfg.sources.ecoforestiere_couche,
                            columns=["cl_age", "cl_dens", "type_couv"]).to_crs(crs)
        log.info("  %d polygones", len(gdf))

    with utils.log_step("Rasterisation âge / densité / couvert", log):
        age = eco.rasteriser_champ(gdf, "cl_age", eco.age_ordinal, transform, w, h, fill=255)
        dens = eco.rasteriser_champ(gdf, "cl_dens", eco.densite_pct, transform, w, h, fill=255)
        tc = eco.rasteriser_champ(gdf, "type_couv", eco.couvert_code, transform, w, h, fill=255)
    del gdf
    dans = tc != 255                                   # emprise de la carte écoforestière

    rapport["age_5m"] = _ecrire_bande(sorties["age_5m"], age, dans, transform, crs)       # var 7
    rapport["densite_5m"] = _ecrire_bande(sorties["densite_5m"], dens, dans, transform, crs)  # var 8
    del age, dens

    with utils.log_step("Proportion feuillu/mélangé (focal)", log):   # var 4
        n_fm = round(cfg.focal.feuillu_m / res)
        is_fm = ((tc == 1) | (tc == 2)).astype("float32")
        prop = uniform_filter(is_fm, size=n_fm, mode="constant", cval=0.0)
        rapport["prop_feuillu_5m"] = _ecrire_bande(sorties["prop_feuillu_5m"], prop, dans,
                                                   transform, crs)
        del is_fm, prop

    with utils.log_step("Densité de lisière forêt-ouvert (focal)", log):   # var 3
        n_lis = round(cfg.focal.lisiere_m / res)
        forest = (tc >= 1) & (tc <= 3)
        edge = forest & binary_dilation(tc == 0)       # forêt touchant du milieu ouvert
        lisiere = uniform_filter(edge.astype("float32"), size=n_lis, mode="constant", cval=0.0)
        rapport["lisiere_5m"] = _ecrire_bande(sorties["lisiere_5m"], lisiere, dans, transform, crs)
        del forest, edge, lisiere
    del tc


def sous_etape_4_routes_mh(cfg: Config, rapport: dict) -> None:
    """Densité de routes (var 5) et distance aux milieux humides (var 9)."""
    from rasterio.features import rasterize
    from scipy.ndimage import uniform_filter

    interim = Path(cfg.chemins.interim)
    _, (_, transform, w, h) = grille_reference(cfg)
    crs = str(cfg.zone_etude.crs)
    res = cfg.zone_etude.resolution_m

    # --- var 5 : densité de routes (km/km², focal routes_m) ---
    sortie_r = interim / "densite_routes_5m.tif"
    if sortie_r.exists():
        log.info("%s déjà présent → réutilisation", sortie_r)
    else:
        with utils.log_step("Routes : lecture zone + reprojection", log):
            routes = utils.lire_vecteur_zone(cfg.sources.routes, cfg.zone_etude.gpkg, crs,
                                             buffer_m=cfg.focal.routes_m)
            routes.to_file(interim / "routes_zone.gpkg", driver="GPKG")   # §3.5 (attributs conservés)
            log.info("  %d tronçons de route dans la zone", len(routes))
        with utils.log_step("Routes : rasterisation + densité (focal)", log):
            presence = rasterize(((g, 1) for g in routes.geometry if g is not None),
                                 out_shape=(h, w), transform=transform, fill=0,
                                 all_touched=True, dtype="uint8")
            n = round(cfg.focal.routes_m / res)
            # densité km/km² : somme des cellules-route dans la fenêtre × longueur / aire (= 1 km²)
            somme = uniform_filter(presence.astype("float32"), size=n, mode="constant") * n * n
            densite = (somme * res / 1000.0).astype("float32")
            utils.write_cog(sortie_r, densite, transform, crs=crs, nodata=None)
    rapport["densite_routes_5m"] = str(sortie_r)

    # --- var 9 : distance au milieu humide le plus proche (m) ---
    sortie_mh = interim / "distance_mh_5m.tif"
    if sortie_mh.exists():
        log.info("%s déjà présent → réutilisation", sortie_mh)
    else:
        with utils.log_step("Milieux humides : lecture zone + reprojection", log):
            mh = utils.lire_vecteur_zone(cfg.sources.milieux_humides, cfg.zone_etude.gpkg, crs,
                                         couche=cfg.sources.milieux_humides_couche, buffer_m=2000)
            mh.to_file(interim / "mh_zone.gpkg", driver="GPKG")           # §3.5 (CLASSE/TYPE/CONFIANCE)
            log.info("  %d polygones de milieu humide dans la zone", len(mh))
        with utils.log_step("Milieux humides : rasterisation + distance euclidienne", log):
            mask = rasterize(((g, 1) for g in mh.geometry if g is not None),
                             out_shape=(h, w), transform=transform, fill=0,
                             dtype="uint8").astype(bool)
            distance = utils.distance_euclidienne(mask, res)
            utils.write_cog(sortie_mh, distance, transform, crs=crs, nodata=None)
    rapport["distance_mh_5m"] = str(sortie_mh)


def sous_etape_5_stack(cfg: Config, rapport: dict) -> None:
    """Assemble les 10 bandes (ordre §2) en un stack COG : nodata en union, MHC clampé ≥ 0."""
    import rasterio
    import rasterio.shutil

    interim = Path(cfg.chemins.interim)
    _, (crs, transform, w, h) = grille_reference(cfg)
    sortie = Path(cfg.chemins.processed) / "stack_5m.tif"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    tmp = interim / "_stack_tmp.tif"

    srcs = [rasterio.open(interim / f"{nom}.tif") for nom, _ in BANDES_STACK]
    try:
        with utils.log_step("Assemblage fenêtré des 10 bandes", log):
            profil = dict(driver="GTiff", width=w, height=h, count=len(BANDES_STACK),
                          dtype="float32", crs=crs, transform=transform, nodata=float("nan"),
                          tiled=True, blockxsize=512, blockysize=512, compress="DEFLATE",
                          BIGTIFF="YES")
            with rasterio.open(tmp, "w", **profil) as dst:
                for i, (_, desc) in enumerate(BANDES_STACK, 1):
                    dst.set_band_description(i, desc)
                for win in utils.iter_windows(w, h, 2048):
                    pile = np.stack([s.read(1, window=win) for s in srcs])   # (10, wh, ww)
                    pile[0] = np.clip(pile[0], 0.0, None)                     # MHC ≥ 0
                    union_nan = ~np.isfinite(pile).all(axis=0)               # nodata en union
                    pile[:, union_nan] = np.nan
                    dst.write(pile.astype("float32"), window=win)
    finally:
        for s in srcs:
            s.close()
    with utils.log_step("Conversion COG", log):
        rasterio.shutil.copy(tmp, sortie, driver="COG", compress="DEFLATE",
                             blocksize=512, BIGTIFF="IF_SAFER")
    tmp.unlink(missing_ok=True)
    rapport["stack_5m"] = str(sortie)
    rapport["bandes"] = [desc for _, desc in BANDES_STACK]
    log.info("Stack écrit : %s (%d bandes)", sortie, len(BANDES_STACK))


def main() -> None:
    cfg = load_config_from_cli()
    utils.setup_logging("01_predictors", log_dir=f"{cfg.chemins.outputs}/logs")
    rapport: dict = {}
    with utils.log_step("Sous-étape 1 — LiDAR (MHC, élévation)", log):
        sous_etape_1_lidar(cfg, rapport)
    with utils.log_step("Sous-étape 2 — TWI", log):
        sous_etape_2_twi(cfg, rapport)
    with utils.log_step("Sous-étape 3 — Écoforestière", log):
        sous_etape_3_ecoforestiere(cfg, rapport)
    with utils.log_step("Sous-étape 4 — Routes + milieux humides", log):
        sous_etape_4_routes_mh(cfg, rapport)
    with utils.log_step("Sous-étape 5 — Assemblage du stack", log):
        sous_etape_5_stack(cfg, rapport)
    utils.ecrire_rapport_json("01_predictors", rapport, log_dir=f"{cfg.chemins.outputs}/logs")


if __name__ == "__main__":
    main()
