"""J2 — Composite de température de surface estivale (LST) via STAC (Landsat C2 L2).

Interroge Microsoft Planetary Computer pour les scènes Landsat niveau-2 sur la zone
d'étude, aux mois d'été de toutes les années eBird, masque nuages/ombres (`QA_PIXEL`),
convertit la bande `ST_B10` en °C, réduit temporellement (médiane des scènes claires)
puis rééchantillonne sur la grille 5 m du stack. Voir CLAUDE.md §3.1 et §3.7.

Sortie : `data/interim/lst_estival_5m.tif` (+ QC : n scènes claires, % manquant).

Les fonctions « pures » (conversion, masque, composite) travaillent sur des tableaux
numpy et sont testées sans réseau ; l'orchestration STAC/xarray importe les libs lourdes
paresseusement.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np

import utils

log = logging.getLogger("02_climate_stac")

# ── Constantes Landsat C2 L2, bande ST_B10 (voir CLAUDE.md §3.1) ─────────────
ST_SCALE = 0.00341802
ST_OFFSET = 149.0
KELVIN = 273.15
# QA_PIXEL — bits à rejeter : fill(0), dilated cloud(1), cirrus(2), cloud(3), cloud shadow(4)
QA_BITS_REJET = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)  # = 31
COLLECTION = "landsat-c2-l2"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
# Bande thermique lwir11 = capteur TIRS de Landsat 8/9 (2013+). Pour des années eBird
# antérieures à 2013, il faudrait Landsat 5/7 (asset `lwir`) — hors périmètre pour l'instant.
PLATEFORMES = ["landsat-8", "landsat-9"]


# ── Fonctions pures (testées sans réseau) ───────────────────────────────────

def st_b10_en_celsius(dn: np.ndarray) -> np.ndarray:
    """Convertit les valeurs brutes `ST_B10` (DN) en °C ; DN==0 (fill) → NaN."""
    dn = np.asarray(dn, dtype="float64")
    celsius = dn * ST_SCALE + ST_OFFSET - KELVIN
    return np.where(dn > 0, celsius, np.nan)


def masque_clair_qa(qa: np.ndarray) -> np.ndarray:
    """Masque booléen des pixels utilisables (True) depuis `QA_PIXEL`.

    Clair si (a) le pixel n'est pas *fill* (`qa != 0` : la donnée Landsat réelle vaut ≥64
    pour un pixel clair, et odc remplit à 0 les lectures d'assets échouées → à exclure) et
    (b) aucun bit fill/dilated-cloud/cirrus/cloud/shadow n'est actif.
    """
    qa = np.asarray(qa).astype("uint16")
    return (qa != 0) & ((qa & QA_BITS_REJET) == 0)


def composite_lst(dn: np.ndarray, qa: np.ndarray,
                  reducteur: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """Composite temporel de LST (°C) à partir d'un cube (temps, y, x).

    Retourne (`lst` 2D °C float32, `n_clair` 2D int16 = nombre de scènes claires/pixel).
    """
    clair = masque_clair_qa(qa)
    celsius = np.where(clair, st_b10_en_celsius(dn), np.nan)
    n_clair = np.isfinite(celsius).sum(axis=0).astype("int16")
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():          # nanmedian sur tranche 100 % NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            reduce_fn = np.nanmedian if reducteur == "median" else np.nanmean
            lst = reduce_fn(celsius, axis=0)
    return lst.astype("float32"), n_clair


# ── Géométrie / grille (pur sauf lecture du gpkg) ───────────────────────────

def emprise_bbox_4326(gpkg: str | Path) -> tuple[float, float, float, float]:
    """Bounding box (minx, miny, maxx, maxy) de la zone d'étude en EPSG:4326."""
    import geopandas as gpd
    minx, miny, maxx, maxy = gpd.read_file(gpkg).to_crs(4326).total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def geobox_zone(cfg, resolution_native: float):
    """GeoBox (grille native, CRS du projet) couvrant l'emprise de la zone d'étude."""
    import geopandas as gpd
    from odc.geo.geobox import GeoBox
    bornes = gpd.read_file(cfg.zone_etude.gpkg).to_crs(cfg.zone_etude.crs).total_bounds
    return GeoBox.from_bbox(tuple(float(b) for b in bornes), crs=cfg.zone_etude.crs,
                            resolution=resolution_native, tight=True)


def bbox_overlap(a, b) -> bool:
    """True si deux bboxes (minx, miny, maxx, maxy) se chevauchent."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def items_intersectant(items, bbox4326):
    """Sous-ensemble des items STAC dont l'emprise (4326) recouvre `bbox4326`."""
    return [it for it in items if bbox_overlap(it.bbox, bbox4326)]


# ── Checkpoint / reprise par tuile ──────────────────────────────────────────

def chemin_tuile(dossier: str | Path, row: int, col: int) -> Path:
    """Chemin du fichier de checkpoint pour la tuile (row, col)."""
    return Path(dossier) / f"lst_r{row:05d}_c{col:05d}.tif"


def tuiles_a_traiter(width: int, height: int, tuile_px: int, dossier: str | Path):
    """Itère (row, col, w, h, chemin, deja_fait) sur les tuiles de la grille.

    `deja_fait` = True si le checkpoint existe déjà (→ reprise, tuile sautée).
    L'écriture atomique de `ecrire_tuile` garantit qu'un fichier présent est complet.
    """
    for win in utils.iter_windows(width, height, tuile_px):
        row, col = int(win.row_off), int(win.col_off)
        path = chemin_tuile(dossier, row, col)
        yield row, col, int(win.width), int(win.height), path, path.exists()


def ecrire_tuile(path: str | Path, lst: np.ndarray, n_clair: np.ndarray,
                 transform, crs) -> Path:
    """Écrit une tuile 2 bandes (LST °C, n_clair) — GTiff, écriture atomique (tmp→rename)."""
    import rasterio
    lst = np.asarray(lst, dtype="float32")
    n_clair = np.asarray(n_clair, dtype="float32")
    h, w = lst.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    profile = dict(driver="GTiff", height=h, width=w, count=2, dtype="float32",
                   crs=crs, transform=transform, nodata=float("nan"),
                   compress="DEFLATE", tiled=True)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(lst, 1)
        dst.write(n_clair, 2)
        dst.set_band_description(1, "lst_c")
        dst.set_band_description(2, "n_clair")
    tmp.replace(path)                                # rename atomique = checkpoint validé
    return path


def composite_tuile(items, geobox, reducteur: str = "median"):
    """Charge le cube d'une tuile (odc-stac, eager), masque + réduit → (lst, n_clair)."""
    import odc.stac
    import planetary_computer
    # fail_on_error=False : un asset corrompu/illisible (fréquent sur l'archive) est ignoré
    # et rempli à 0 → exclu par masque_clair_qa (qa==0) au lieu de faire planter la tuile.
    cube = odc.stac.load(items, bands=["lwir11", "qa_pixel"], geobox=geobox,
                         groupby="solar_day", patch_url=planetary_computer.sign,
                         fail_on_error=False)
    dn = np.asarray(cube["lwir11"].values)
    qa = np.asarray(cube["qa_pixel"].values)
    return composite_lst(dn, qa, reducteur)


def mosaiquer_tuiles(paths):
    """Assemble les tuiles de checkpoint en une mosaïque 2 bandes (band, y, x, NaN=nodata)."""
    import rioxarray  # noqa: F401  (enregistre l'accès .rio)
    from rioxarray.merge import merge_arrays
    arrs = [rioxarray.open_rasterio(p, masked=True) for p in paths]
    return merge_arrays(arrs)


# ── Recherche STAC (réseau, imports paresseux) ──────────────────────────────


def annees_cible(cfg) -> list[int]:
    """Années du composite : `cfg.climat_stac.annees` si fixé, sinon repli documenté."""
    annees = cfg.climat_stac.annees
    if annees:
        return list(annees)
    fin = dt.date.today().year - 1
    repli = list(range(fin - 9, fin + 1))        # 10 dernières années par défaut
    log.warning("climat_stac.annees non fixé → repli %d–%d ; renseigner après J4 (années eBird).",
                repli[0], repli[-1])
    return repli


def intervalles_datetime(annees: list[int], mois: list[int]) -> list[str]:
    """Intervalles STAC 'AAAA-MM-01/AAAA-MM-JJ' pour chaque année × plage de mois."""
    m0, m1 = min(mois), max(mois)
    fin_jour = (dt.date(2000, m1 % 12 + 1, 1) - dt.timedelta(days=1)).day if m1 != 12 else 31
    return [f"{a}-{m0:02d}-01/{a}-{m1:02d}-{fin_jour:02d}" for a in annees]


def rechercher_items(bbox, annees, mois, cloud_max: float, plateformes: list[str] | None = None):
    """Recherche les scènes Landsat 8/9 C2 L2 (items STAC non signés ; signés à la lecture)."""
    import pystac_client
    client = pystac_client.Client.open(PC_STAC)
    plateformes = plateformes or PLATEFORMES
    items = []
    for intervalle in intervalles_datetime(annees, mois):
        recherche = client.search(collections=[COLLECTION], bbox=bbox, datetime=intervalle,
                                   query={"eo:cloud_cover": {"lt": cloud_max},
                                          "platform": {"in": plateformes}})
        lot = list(recherche.items())
        log.info("STAC %s : %d scènes (nuages < %g%%, %s)",
                 intervalle, len(lot), cloud_max, "/".join(plateformes))
        items.extend(lot)
    return items


def construire_composite_lst(cfg, sortie: str | Path | None = None,
                             dossier_tuiles: str | Path | None = None) -> Path:
    """Pipeline J2 : requête STAC → composite médian **par tuile** (checkpoint/reprise)
    → mosaïque → rééchantillonnage 5 m → COG + QC.

    Chaque tuile est calculée puis **écrite immédiatement** (checkpoint atomique). Une
    relance saute les tuiles déjà présentes : une coupure ne coûte que la tuile en cours.
    La médiane étant calculée pixel par pixel, le tuilage spatial donne un résultat
    identique à une médiane globale, tout en bornant la mémoire (une tuile à la fois).
    """
    import rioxarray  # noqa: F401  (enregistre l'accès .rio)
    from rasterio.enums import Resampling
    from tqdm import tqdm

    utils.configurer_gdal_cloud()
    interim = cfg.chemins.interim
    sortie = Path(sortie or f"{interim}/lst_estival_5m.tif")
    dossier_tuiles = Path(dossier_tuiles or f"{interim}/lst_tiles")
    res_native = cfg.climat_stac.resolution_native_m
    res_finale = cfg.zone_etude.resolution_m
    tuile_px = cfg.climat_stac.tuile_px
    reducteur = cfg.climat_stac.reducteur
    annees = annees_cible(cfg)
    rapport = {"annees": annees, "mois": cfg.climat_stac.mois, "reducteur": reducteur,
               "resolution_native_m": res_native, "tuile_px": tuile_px}

    with utils.log_step("Requête STAC Landsat", log):
        bbox = emprise_bbox_4326(cfg.zone_etude.gpkg)
        items = rechercher_items(bbox, annees, cfg.climat_stac.mois,
                                 cfg.climat_stac.couverture_nuageuse_max)
        rapport["n_scenes"] = len(items)
    if not items:
        raise RuntimeError("Aucune scène Landsat trouvée pour la zone/période.")

    gb = geobox_zone(cfg, res_native)
    with utils.log_step("Composite par tuiles (checkpoint/reprise)", log):
        taches = list(tuiles_a_traiter(gb.width, gb.height, tuile_px, dossier_tuiles))
        chemins_tuiles = [t[4] for t in taches]
        n_reprise = 0
        for row, col, w, h, path, deja_fait in tqdm(taches, desc="tuiles", unit="tuile"):
            if deja_fait:
                n_reprise += 1
                log.debug("tuile r%d c%d — reprise (checkpoint présent)", row, col)
                continue
            sub = gb[row:row + h, col:col + w]
            items_sub = items_intersectant(items, tuple(sub.geographic_extent.boundingbox))
            if items_sub:
                lst, n_clair = composite_tuile(items_sub, sub, reducteur)
            else:                                    # aucune scène sur la tuile → NaN plein
                lst = np.full((h, w), np.nan, dtype="float32")
                n_clair = np.zeros((h, w), dtype="int16")
            ecrire_tuile(path, lst, n_clair, sub.transform, cfg.zone_etude.crs)
            log.debug("tuile r%d c%d écrite (%d scènes)", row, col, len(items_sub))
        rapport.update(n_tuiles=len(taches), n_tuiles_reprises=n_reprise)

    with utils.log_step(f"Mosaïque + rééchantillonnage {res_native} m → {res_finale} m + COG", log):
        mosaic = mosaiquer_tuiles(chemins_tuiles)
        lst30 = mosaic.isel(band=0).rio.write_crs(cfg.zone_etude.crs)
        n_clair30 = mosaic.isel(band=1).values
        # Comblement des trous du produit ST USGS (fill sur scènes claires, pas des nuages) :
        # interpolation locale au natif 30 m avant rééchantillonnage. Voir CLAUDE.md §3.1.
        rapport["pct_manquant_brut"] = round(float((~np.isfinite(lst30.values)).mean() * 100), 2)
        if cfg.climat_stac.combler_trous:
            vals = lst30.values
            n_avant = int(np.isnan(vals).sum())
            comble = utils.combler_nodata(vals, cfg.climat_stac.comblement_max_px)
            rapport["n_pixels_combles"] = n_avant - int(np.isnan(comble).sum())
            lst30 = lst30.copy(data=comble)
            log.info("Trous ST comblés : %d px (interpolation locale, max %d px natifs)",
                     rapport["n_pixels_combles"], cfg.climat_stac.comblement_max_px)
        lst5 = lst30.rio.reproject(cfg.zone_etude.crs, resolution=res_finale,
                                   resampling=Resampling.bilinear)
        lst5.attrs["long_name"] = "lst_c"            # mono-bande (mosaïque → 2 noms sinon)
        sortie.parent.mkdir(parents=True, exist_ok=True)
        lst5.rio.to_raster(sortie, driver="COG", compress="DEFLATE", blocksize=512)

    # QC : statistiques du composite et de la couverture claire (lues une seule fois)
    vals = lst5.values
    finite = np.isfinite(vals)
    rapport.update(
        sortie=str(sortie),
        pct_manquant=round(float((~finite).mean() * 100), 2),
        lst_c_min=round(float(np.nanmin(vals)), 2) if finite.any() else None,
        lst_c_moy=round(float(np.nanmean(vals)), 2) if finite.any() else None,
        lst_c_max=round(float(np.nanmax(vals)), 2) if finite.any() else None,
        n_clair_median=int(np.nanmedian(n_clair30)),
    )
    utils.ecrire_rapport_json("02_climate_stac", rapport, log_dir=f"{cfg.chemins.commun}/logs")
    log.info("Composite LST écrit : %s (%.1f%% manquant, %d/%d tuiles reprises)",
             sortie, rapport["pct_manquant"], n_reprise, len(taches))
    return sortie


def main() -> None:
    from config import load_config_from_cli
    cfg = load_config_from_cli()
    utils.setup_logging("02_climate_stac", log_dir=f"{cfg.chemins.commun}/logs")
    construire_composite_lst(cfg)


if __name__ == "__main__":
    main()
