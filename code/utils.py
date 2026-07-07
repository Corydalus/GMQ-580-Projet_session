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
