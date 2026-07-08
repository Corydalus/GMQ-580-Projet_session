"""Fonctions partagées du pipeline SDM Engoulevent : config, journalisation,
parallélisation Dask, I/O raster (COG), statistiques focales, fenêtrage, MESS.

Les imports lourds (rasterio, scipy, dask) sont **paresseux** (à l'intérieur des
fonctions) pour qu'un simple `import utils` reste léger et testable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Conventions globales (valeurs par défaut ; surchargées par config.yaml) ──
CRS = "EPSG:32198"        # NAD83 / Québec Lambert — unique pour rasters et vecteurs
RESOLUTION_M = 5          # repli 10 m si RAM insuffisante
RANDOM_STATE = 42         # reproductibilité partout
EBIRD_BUFFER_M = 30       # buffer d'extraction des covariables (précision GPS eBird)
COG_PROFILE = dict(driver="COG", compress="DEFLATE", blocksize=512)


def set_gdal_cache(mb: int = 512) -> None:
    """Fixe GDAL_CACHEMAX (à appeler dans chaque worker multiprocessing)."""
    os.environ["GDAL_CACHEMAX"] = str(mb)


def configurer_gdal_cloud() -> None:
    """Réglages GDAL pour la lecture de COG signés sur le cloud (STAC / Planetary Computer)."""
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
    os.environ.setdefault("VSI_CACHE", "TRUE")


# ── Journalisation & rapports (voir CLAUDE.md §7) ───────────────────────────

def setup_logging(script: str, verbose: bool = False,
                  log_dir: str | Path = "outputs/logs") -> tuple[logging.Logger, Path]:
    """Configure le logging (console + fichier horodaté). Retourne (logger, chemin)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{script}_{datetime.now():%Y%m%d-%H%M%S}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):          # évite les handlers dupliqués
        root.removeHandler(h)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)

    log = logging.getLogger(script)
    log.info("Journalisation → %s", log_path)
    return log, log_path


@contextmanager
def log_step(nom: str, logger: logging.Logger | None = None):
    """Gestionnaire de contexte chronométré : journalise début, durée, succès/échec."""
    log = logger or logging.getLogger("pipeline")
    log.info("▶ %s — début", nom)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception("✗ %s — échec après %.1f s", nom, time.perf_counter() - t0)
        raise
    log.info("✔ %s — terminé en %.1f s", nom, time.perf_counter() - t0)


def ecrire_rapport_json(script: str, rapport: dict,
                        log_dir: str | Path = "outputs/logs") -> Path:
    """Écrit le rapport de fin d'exécution (compteurs, durées, QC) en JSON."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{script}_rapport.json"
    contenu = {"script": script,
               "horodatage": datetime.now().isoformat(timespec="seconds"),
               **rapport}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(contenu, f, ensure_ascii=False, indent=2, default=str)
    return path


# ── Parallélisation Dask (voir CLAUDE.md §3.7) ──────────────────────────────

def get_dask_client(cfg=None, **overrides):
    """Crée un `LocalCluster` + `Client` selon le bloc `calcul` du config.

    `cfg` : objet Config (ou None → valeurs par défaut). `overrides` : n_workers,
    threads_per_worker, memory_limit, dashboard. À fermer (`client.close()`) ou
    à utiliser en `with`.
    """
    from dask.distributed import Client, LocalCluster

    calc = getattr(cfg, "calcul", None)
    params = dict(
        n_workers=getattr(calc, "n_workers", None),
        threads_per_worker=getattr(calc, "threads_per_worker", 2),
        memory_limit=getattr(calc, "memory_limit", "auto"),
        dashboard=getattr(calc, "dashboard", True),
        processes=getattr(calc, "processes", True),
    )
    params.update(overrides)
    cluster = LocalCluster(
        n_workers=params["n_workers"],
        threads_per_worker=params["threads_per_worker"],
        memory_limit=params["memory_limit"],
        processes=params["processes"],
        dashboard_address=":8787" if params["dashboard"] else None,
    )
    return Client(cluster)


# ── Statistiques focales (voir CLAUDE.md §3.7 : map_overlap en production) ───

def focal_mean(arr: np.ndarray, size: int) -> np.ndarray:
    """Moyenne focale (fenêtre carrée `size`), en ignorant les NaN."""
    from scipy.ndimage import uniform_filter
    arr = np.asarray(arr, dtype="float64")
    valid = np.isfinite(arr)
    somme = uniform_filter(np.where(valid, arr, 0.0), size=size, mode="constant", cval=0.0)
    fraction = uniform_filter(valid.astype("float64"), size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = somme / fraction
    out[fraction == 0] = np.nan
    return out.astype("float32")


def focal_sum(arr: np.ndarray, size: int) -> np.ndarray:
    """Somme focale (fenêtre carrée `size`) — base des densités (longueur/compte)."""
    from scipy.ndimage import uniform_filter
    arr = np.asarray(arr, dtype="float64")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return (uniform_filter(arr, size=size, mode="constant", cval=0.0) * size * size).astype("float32")


def focal_proportion(mask: np.ndarray, size: int) -> np.ndarray:
    """Proportion de cellules vraies dans la fenêtre (p. ex. couvert forestier)."""
    from scipy.ndimage import uniform_filter
    mask = np.asarray(mask, dtype="float64")
    return uniform_filter(mask, size=size, mode="constant", cval=0.0).astype("float32")


# ── I/O raster ──────────────────────────────────────────────────────────────

def write_cog(path: str | Path, array: np.ndarray, transform, crs: str = CRS,
              nodata: float | None = None) -> Path:
    """Écrit un COG DEFLATE (profil `COG_PROFILE`). `array` 2D ou (bandes, H, W)."""
    import rasterio
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    count, height, width = array.shape
    profile = dict(COG_PROFILE)
    profile.update(width=width, height=height, count=count,
                   dtype=str(array.dtype), crs=crs, transform=transform)
    if nodata is not None:
        profile["nodata"] = nodata
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
    return path


def iter_windows(width: int, height: int, tile: int = 1024):
    """Itère des fenêtres `rasterio` (tuilage) — lecture/écriture fenêtrée."""
    from rasterio.windows import Window
    for row in range(0, height, tile):
        for col in range(0, width, tile):
            yield Window(col, row, min(tile, width - col), min(tile, height - row))


def reproject_to_grid(src_array: np.ndarray, src_transform, src_crs,
                      dst_transform, dst_crs, dst_shape,
                      resampling: str = "bilinear",
                      src_nodata: float | None = None) -> np.ndarray:
    """Rééchantillonne/reprojette un tableau source vers une grille cible définie."""
    import rasterio.warp as warp
    from rasterio.enums import Resampling
    dst = np.empty(dst_shape, dtype="float32")
    warp.reproject(
        source=np.asarray(src_array),
        destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        src_nodata=src_nodata,
        resampling=getattr(Resampling, resampling),
    )
    return dst


def wbt_run(tool: str, work_dir: str | Path | None = None, **params) -> str:
    """Exécute un outil WhiteboxTools via son binaire (robuste aux espaces du chemin).

    Télécharge le binaire au 1er appel si absent, le rend exécutable, puis lance
    `whitebox_tools --run=<tool> --<param>=<valeur> …`. Lève RuntimeError si échec.
    Appel direct au binaire pour contourner un bug du wrapper Python avec les chemins
    contenant des espaces.
    """
    import stat
    import subprocess

    import whitebox
    exe = Path(whitebox.__file__).parent / "WBT" / "whitebox_tools"
    if not exe.exists():
        whitebox.WhiteboxTools()                 # déclenche le téléchargement du binaire
    os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    args = [str(exe), f"--run={tool}"]
    if work_dir is not None:
        args.append(f"--wd={work_dir}")
    args += [f"--{k}={v}" for k, v in params.items()]
    args.append("-v=false")
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"WhiteboxTools {tool} a échoué :\n{(res.stderr or res.stdout)[-500:]}")
    return res.stdout


def lire_grille_reference(path: str | Path):
    """Retourne (crs, transform, width, height) d'un raster servant de grille de référence."""
    import rasterio
    with rasterio.open(path) as ds:
        return ds.crs, ds.transform, ds.width, ds.height


def agreger_tuiles_vers_grille(tuiles, ref_crs, ref_transform, width: int, height: int,
                               resampling: str = "average",
                               dst_nodata: float = float("nan")) -> np.ndarray:
    """Reprojette/agrège des tuiles raster (autre CRS/résolution) sur une grille de référence.

    Chaque tuile est lue en flux (jamais toute la mosaïque en RAM) et reprojetée dans sa
    fenêtre de la grille cible (`resampling`, ex. 'average' pour agréger 1 m → 5 m). Les
    tuiles ne se chevauchant pas, on écrit les valeurs valides dans la sortie accumulée.
    """
    import math

    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    out = np.full((height, width), dst_nodata, dtype="float32")
    for fp in tuiles:
        with rasterio.open(fp) as src:
            left, bottom, right, top = transform_bounds(src.crs, ref_crs, *src.bounds,
                                                        densify_pts=21)
            inv = ~ref_transform                      # coords carte → indices (col, row)
            c0, r0 = inv * (left, top)
            c1, r1 = inv * (right, bottom)
            col_off = max(0, math.floor(min(c0, c1)))
            row_off = max(0, math.floor(min(r0, r1)))
            col_end = min(width, math.ceil(max(c0, c1)))
            row_end = min(height, math.ceil(max(r0, r1)))
            if col_end <= col_off or row_end <= row_off:
                continue                              # tuile hors grille
            win = Window(col_off, row_off, col_end - col_off, row_end - row_off)
            dst = np.full((int(win.height), int(win.width)), dst_nodata, dtype="float32")
            reproject(source=rasterio.band(src, 1), destination=dst,
                      dst_transform=window_transform(win, ref_transform), dst_crs=ref_crs,
                      src_nodata=src.nodata, dst_nodata=dst_nodata,
                      resampling=getattr(Resampling, resampling), num_threads=2)
            sub = out[row_off:row_end, col_off:col_end]
            valide = np.isfinite(dst) if np.isnan(dst_nodata) else (dst != dst_nodata)
            sub[valide] = dst[valide]
    return out


def combler_nodata(arr: np.ndarray, max_dist: int, smoothing: int = 0) -> np.ndarray:
    """Comble les NaN par interpolation locale (IDW, `rasterio.fill.fillnodata`).

    `max_dist` : distance de recherche max (pixels) ; les trous plus profonds restent NaN.
    Retourne un float32 ; les pixels non comblés (hors portée) conservent NaN.
    """
    from rasterio.fill import fillnodata
    arr = np.asarray(arr, dtype="float32")
    valide = np.isfinite(arr)
    image = np.where(valide, arr, 0.0).astype("float32")   # fillnodata n'aime pas les NaN
    rempli = fillnodata(image, mask=valide.astype("uint8"),
                        max_search_distance=float(max_dist), smoothing_iterations=smoothing)
    rempli = np.asarray(rempli, dtype="float32")
    non_comble = ~np.isfinite(arr) & (rempli == 0.0) & ~valide   # hors portée → reste NaN
    rempli[non_comble] = np.nan
    return rempli


def lire_vecteur_zone(chemin: str | Path, zone_gpkg: str | Path, crs_cible,
                      couche: str | None = None, buffer_m: float = 0.0):
    """Lit une couche vectorielle filtrée par l'emprise de la zone (bbox côté driver, §3.4).

    La bbox est calculée dans le CRS *source* de la couche (jamais toute la province en
    RAM), puis le résultat clippé est reprojeté vers `crs_cible`. `buffer_m` élargit la
    bbox (unités du CRS source) pour les effets de bord focaux.
    """
    import geopandas as gpd
    import pyogrio
    crs_src = pyogrio.read_info(str(chemin), layer=couche)["crs"]
    zone = gpd.read_file(zone_gpkg).to_crs(crs_src)
    minx, miny, maxx, maxy = zone.total_bounds
    bbox = (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)
    gdf = gpd.read_file(chemin, layer=couche, bbox=bbox)
    return gdf.to_crs(crs_cible)


def distance_euclidienne(mask: np.ndarray, resolution: float) -> np.ndarray:
    """Distance euclidienne (m) de chaque cellule au pixel `True` le plus proche.

    `mask` True = présence (p. ex. milieu humide). Tout-False → tableau de NaN.
    """
    from scipy.ndimage import distance_transform_edt
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, np.nan, dtype="float32")
    return (distance_transform_edt(~mask) * resolution).astype("float32")


# ── MESS — Multivariate Environmental Similarity Surface (utilisée à J6) ─────

def mess(proj: np.ndarray, ref) -> np.ndarray:
    """MESS (Elith et al. 2010) : similarité aux conditions d'entraînement.

    `proj` : (n_var, H, W) valeurs à projeter. `ref` : séquence de n_var vecteurs
    de référence (valeurs d'entraînement). Retourne (H, W) ; négatif = extrapolation.
    """
    proj = np.asarray(proj, dtype="float64")
    n_var = proj.shape[0]
    plat = proj.reshape(n_var, -1)
    sim = np.full(plat.shape, np.nan)
    for i in range(n_var):
        v = np.asarray(ref[i], dtype="float64")
        v = np.sort(v[np.isfinite(v)])
        vmin, vmax = v[0], v[-1]
        rng = (vmax - vmin) or np.nan          # constante → NaN (variable ignorée)
        p = plat[i]
        f = np.searchsorted(v, p, side="left") / v.size * 100.0  # % de réf < p
        with np.errstate(invalid="ignore", divide="ignore"):
            sim[i] = np.where(f == 0, (p - vmin) / rng * 100.0,
                     np.where(f <= 50, 2.0 * f,
                     np.where(f < 100, 2.0 * (100.0 - f),
                              (vmax - p) / rng * 100.0)))
    return np.nanmin(sim, axis=0).reshape(proj.shape[1:]).astype("float32")
