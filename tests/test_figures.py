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


def test_fig_carte_routes_hotspots_legende(tmp_path):
    """Carte avec routes principales + emprise + hotspots → exerce légende et liseré (path effects)."""
    import geopandas as gpd
    from shapely.geometry import LineString, box
    m = _charger_figures()
    raster = tmp_path / "r.tif"
    _mini_raster(raster)  # extent : x [-300000, -299750], y [199750, 200000]
    zone = gpd.GeoDataFrame(geometry=[box(-300000, 199750, -299750, 200000)], crs="EPSG:32198")
    routes = gpd.GeoDataFrame(
        {"ClsRte": ["Autoroute"]},
        geometry=[LineString([(-300000, 199800), (-299800, 199950)])], crs="EPSG:32198")
    hotspots = gpd.GeoDataFrame(
        {"rang": [1]}, geometry=[box(-299950, 199800, -299900, 199850)], crs="EPSG:32198")
    out = tmp_path / "carte_full.png"
    m.fig_carte(str(raster), "titre", "valeur", str(out), vmin=0, vmax=1,
                zone=zone, routes=routes, hotspots=hotspots)
    assert _existe_non_vide(out)


def test_fig_points_ebird(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    m = _charger_figures()
    zone = gpd.GeoDataFrame(geometry=[box(-300000, 190000, -290000, 200000)], crs="EPSG:32198")
    rng = np.random.default_rng(5)
    couches = [
        dict(x=rng.uniform(-300000, -290000, 40), y=rng.uniform(190000, 200000, 40),
             couleur="#4575b4", taille=4, alpha=0.3, label="sans détection"),
        dict(x=rng.uniform(-300000, -290000, 8), y=rng.uniform(190000, 200000, 8),
             couleur="#d73027", edge="white", label="présences"),
    ]
    out = tmp_path / "pts.png"
    m.fig_points_ebird(zone, couches, str(out), "titre", fond=False)  # fond=False : pas de réseau
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


# ── Figures SHAP & diagnostic élévation ──────────────────────────────────────

_FEATURES_HABITAT = ["MHC_hauteur_canopee", "TWI", "densite_lisiere", "prop_feuillu_melange",
                     "densite_routes", "LST_estivale", "classe_age", "densite_peuplement",
                     "distance_milieu_humide", "elevation"]


def _mini_shap_npz(path, n=60):
    rng = np.random.default_rng(3)
    np.savez_compressed(
        path, values=rng.normal(0, 0.05, (n, len(_FEATURES_HABITAT))).astype("float32"),
        data=rng.random((n, len(_FEATURES_HABITAT))).astype("float32"),
        presence=(rng.random(n) < 0.2).astype(int), base_value=np.float64(0.1),
        features=np.array(_FEATURES_HABITAT))


def test_fig_shap(tmp_path):
    pytest.importorskip("shap")
    m = _charger_figures()
    npz = tmp_path / "shap.npz"
    _mini_shap_npz(npz)
    out = tmp_path / "shap.png"
    m.fig_shap(str(npz), str(out))
    assert _existe_non_vide(out)


def test_fig_shap_elevation(tmp_path):
    m = _charger_figures()  # figure manuelle : pas de dépendance à shap au tracé
    npz = tmp_path / "shap.npz"
    _mini_shap_npz(npz)
    out = tmp_path / "shap_elev.png"
    m.fig_shap_elevation(str(npz), str(out))
    assert _existe_non_vide(out)


def test_fig_diagnostic_elevation(tmp_path):
    m = _charger_figures()
    diag = dict(r2_elevation_xy=0.36, corr_elevation_x=0.42, corr_elevation_y=-0.40,
                elevation_sans_coords=dict(importance=0.11, rang=1),
                elevation_avec_coords=dict(importance=0.05, rang=4),
                coords=dict(x=dict(importance=0.07, rang=2), y=dict(importance=0.06, rang=3)),
                auc_habitat=0.86, auc_habitat_xy=0.87)
    out = tmp_path / "diag.png"
    m.fig_diagnostic_elevation(diag, str(out))
    assert _existe_non_vide(out)
