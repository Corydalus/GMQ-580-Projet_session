"""Fonctions partagées du pipeline SDM Engoulevent : I/O, CRS, focal stats, MESS."""

import os

# ── Conventions globales (voir CLAUDE.md §8) ────────────────────────────────
CRS = "EPSG:32198"        # NAD83 / Québec Lambert — unique pour rasters et vecteurs
RESOLUTION_M = 5          # repli 10 m si RAM insuffisante
RANDOM_STATE = 42         # reproductibilité partout
EBIRD_BUFFER_M = 30       # buffer d'extraction des covariables (précision GPS eBird)
COG_PROFILE = dict(driver="COG", compress="DEFLATE", blocksize=512)


def set_gdal_cache(mb: int = 512) -> None:
    """Fixe GDAL_CACHEMAX (à appeler dans chaque worker multiprocessing)."""
    os.environ["GDAL_CACHEMAX"] = str(mb)


# TODO (feat/predictors, feat/model) : focal_mean, focal_density, mess, ...
