"""Encodage de la carte écoforestière MFFP (couche `pee`) pour le stack SDM (J3).

Fonctions pures d'encodage (testées sans données) + rasterisation d'un champ encodé
sur la grille de référence. Variables produites (voir CLAUDE.md §2) :
  - var 7 âge      : `cl_age` → ordinal de maturité 0–4 (in-)équiennes gérées
  - var 8 densité  : `cl_dens` → % de couvert (point milieu de classe)
  - var 4/3        : `type_couv` (F/M/R) → code couvert (base de la prop. feuillue et lisière)
"""

from __future__ import annotations

import numpy as np

# ── Encodage (pur) ───────────────────────────────────────────────────────────

# Ordinal de maturité par 1er terme du code (strate dominante listée en premier).
_MATURITE = {"10": 1, "JIN": 1, "JIR": 1,     # 1 = jeune / régénération
             "30": 2, "50": 2,                # 2 = mature
             "70": 3, "90": 3,                # 3 = âgé
             "120": 4, "VIN": 4, "VIR": 4}    # 4 = vieux / inéquienne âgé
_DENSITE_PCT = {"A": 90, "B": 70, "C": 50, "D": 32}   # % de couvert (point milieu)
_COUVERT = {"F": 1, "M": 2, "R": 3}                   # feuillu / mélangé / résineux


def _premier_terme(code: str) -> str:
    """Premier terme d'un code d'âge (lettre JIN/JIR/VIN/VIR, ou nombre 120/10..90)."""
    if code[:3] in {"JIN", "JIR", "VIN", "VIR"}:
        return code[:3]
    if code.startswith("120"):
        return "120"
    return code[:2]


def age_ordinal(code) -> int:
    """`cl_age` → ordinal de maturité : 0 non-forêt, 1 jeune, 2 mature, 3 âgé, 4 vieux."""
    if code is None or (isinstance(code, float) and np.isnan(code)) or code == "":
        return 0
    return _MATURITE.get(_premier_terme(str(code)), 0)


def densite_pct(code) -> int:
    """`cl_dens` A/B/C/D → % de couvert (point milieu) ; non-forêt → 0."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return 0
    return _DENSITE_PCT.get(str(code), 0)


def couvert_code(type_couv) -> int:
    """`type_couv` → 0 non-forêt, 1 feuillu (F), 2 mélangé (M), 3 résineux (R)."""
    if type_couv is None or (isinstance(type_couv, float) and np.isnan(type_couv)):
        return 0
    return _COUVERT.get(str(type_couv), 0)


# ── Rasterisation d'un champ encodé sur la grille de référence ───────────────

def rasteriser_champ(gdf, colonne: str, encoder, transform, width: int, height: int,
                     fill: int = 255, dtype: str = "uint8") -> np.ndarray:
    """Rasterise `encoder(gdf[colonne])` sur la grille cible ; `fill` hors polygones."""
    from rasterio.features import rasterize
    formes = ((geom, encoder(val)) for geom, val in zip(gdf.geometry, gdf[colonne], strict=True)
              if geom is not None)
    return rasterize(formes, out_shape=(height, width), transform=transform,
                     fill=fill, dtype=dtype)
