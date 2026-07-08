"""Tests de `code/05_figures.py` — génération des PNG (J7).

Le module a un préfixe numérique (non importable directement) : on le charge via importlib.
Chaque figure est produite sur des données factices ; on vérifie l'existence + taille non nulle
du PNG. Aucun réseau (fond OSM désactivé : fond=False).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402


def _charger_figures():
    chemin = Path(__file__).resolve().parent.parent / "code" / "05_figures.py"
    spec = importlib.util.spec_from_file_location("figures05", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _existe_non_vide(p) -> bool:
    p = Path(p)
    return p.exists() and p.stat().st_size > 0


def _mini_raster(path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    arr = np.random.default_rng(0).random((50, 50)).astype("float32")
    arr[:5, :5] = np.nan
    prof = dict(driver="GTiff", width=50, height=50, count=1, dtype="float32",
                crs="EPSG:32198", transform=from_origin(-300000, 200000, 5, 5), nodata=float("nan"))
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr, 1)


def test_fig_carte(tmp_path):
    m = _charger_figures()
    raster = tmp_path / "r.tif"
    _mini_raster(raster)
    out = tmp_path / "carte.png"
    m.fig_carte(str(raster), "titre", "valeur", str(out), vmin=0, vmax=1)
    assert _existe_non_vide(out)


def test_fig_carte_mess_centre0_fidele(tmp_path):
    m = _charger_figures()
    raster = tmp_path / "mess.tif"
    _mini_raster(raster)
    out = tmp_path / "mess.png"
    m.fig_carte(str(raster), "MESS", "MESS", str(out), centre0=True, fidele=True)
    assert _existe_non_vide(out)


def test_fig_localisation(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    m = _charger_figures()
    zone = gpd.GeoDataFrame(geometry=[box(-300000, 190000, -290000, 200000)], crs="EPSG:32198")
    out = tmp_path / "loc.png"
    m.fig_localisation(zone, str(out), fond=False)  # fond=False : pas de réseau en test
    assert _existe_non_vide(out)


def test_fig_importance(tmp_path):
    m = _charger_figures()
    imp = [dict(variable="elevation", importance=0.11, ecart_type=0.01),
           dict(variable="TWI", importance=0.02, ecart_type=0.005),
           dict(variable="LST_estivale", importance=0.01, ecart_type=0.002)]
    out = tmp_path / "imp.png"
    m.fig_importance(imp, "Importance", str(out))
    assert _existe_non_vide(out)


def test_fig_pdp(tmp_path):
    m = _charger_figures()
    pdp = {"horodatage": "x",
           "elevation": dict(grille=[0, 1, 2], pd=[0.1, 0.2, 0.3]),
           "TWI": dict(grille=[3, 4, 5], pd=[0.2, 0.25, 0.22])}
    out = tmp_path / "pdp.png"
    m.fig_pdp(pdp, str(out))
    assert _existe_non_vide(out)


def test_fig_metriques_cv(tmp_path):
    m = _charger_figures()
    rapport = {
        "modele_combine": dict(auc_folds=[0.97, 0.98, 0.96], auc_moy=0.97, auc_std=0.01,
                               tss_folds=[0.88, 0.86, 0.90], tss_moy=0.88, tss_std=0.02),
        "modele_habitat": dict(auc_folds=[0.88, 0.80, 0.87], auc_moy=0.85, auc_std=0.03,
                               tss_folds=[0.68, 0.62, 0.68], tss_moy=0.66, tss_std=0.03),
    }
    out = tmp_path / "cv.png"
    m.fig_metriques_cv(rapport, str(out))
    assert _existe_non_vide(out)
