"""Tests de `code/04_predict.py` — prédiction fenêtrée, MESS, hotspots (J6).

Le module a un préfixe numérique (non importable directement) : on le charge via importlib.
Fixtures synthétiques légères — pas de réseau, pas de données réelles.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
import numpy as np  # noqa: E402


def _charger_predict():
    chemin = Path(__file__).resolve().parent.parent / "code" / "04_predict.py"
    spec = importlib.util.spec_from_file_location("predict04", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mini_rf():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.random((200, 14)).astype("float32")
    y = (X[:, 0] + X[:, 9] > 1.0).astype(int)          # signal sur 2 variables
    return RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)


# ── Prédiction : moyenne = predict_proba, et fenêtrée == pleine ──────────────

def test_moyenne_egale_predict_proba():
    m = _charger_predict()
    rf = _mini_rf()
    X = np.random.default_rng(1).random((500, 14)).astype("float32")
    proba, var = m.predire_moyenne_variance(rf, X)
    assert np.allclose(proba, rf.predict_proba(X)[:, 1], atol=1e-6)
    # variance inter-arbres reconstruite à la main
    preds = np.stack([t.predict_proba(X)[:, 1] for t in rf.estimators_])
    assert np.allclose(var, preds.var(axis=0), atol=1e-6)
    assert (var >= 0).all()


def test_prediction_fenetree_egale_pleine():
    m = _charger_predict()
    rf = _mini_rf()
    X = np.random.default_rng(2).random((500, 14)).astype("float32")
    pleine, _ = m.predire_moyenne_variance(rf, X)
    # découpage en « fenêtres » de lignes puis recombinaison → identique (pixels indépendants)
    morceaux = [m.predire_moyenne_variance(rf, X[i:i + 128])[0] for i in range(0, len(X), 128)]
    assert np.allclose(np.concatenate(morceaux), pleine, atol=1e-7)


# ── MESS : extrapolation hors domaine d'entraînement (cas connu) ─────────────

def test_mess_extrapolation():
    import utils
    ref = [np.arange(0.0, 11.0)]                        # 1 variable, entraînement 0..10
    proj = np.array([[[5.0, 20.0, -3.0]]])              # (1 var, 1, 3) : dedans / au-dessus / en-dessous
    out = utils.mess(proj, ref).ravel()
    assert out[0] > 0                                   # 5 dans l'intervalle → similarité positive
    assert out[1] < 0 and out[2] < 0                    # 20 et -3 hors bornes → extrapolation


# ── Hotspots : haute proba + faible effort, cellules surveillées exclues ─────

def _ecrire_proba(path, top_gauche=0.9, top_droit=0.9, fond=0.1):
    """Grille proba 600×600 à 5 m (→ 3×3 cellules 1 km) : deux cellules à 0,9."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    left, top = -300000.0, 200000.0
    arr = np.full((600, 600), fond, dtype="float32")
    arr[0:200, 0:200] = top_gauche                      # cellule (0,0)
    arr[400:600, 400:600] = top_droit                   # cellule (2,2)
    prof = dict(driver="GTiff", width=600, height=600, count=1, dtype="float32",
                crs="EPSG:32198", transform=from_origin(left, top, 5, 5), nodata=float("nan"))
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr, 1)
    return left, top


def test_hotspots(tmp_path, sample_config_dict):
    import polars as pl

    from config import Config
    m = _charger_predict()
    cfg = Config.model_validate(sample_config_dict)
    proba_path = tmp_path / "proba.tif"
    left, top = _ecrire_proba(proba_path)
    # effort : plusieurs checklists dans la cellule (0,0), aucune dans la cellule (2,2)
    xs = [left + 500] * 4
    ys = [top - 500] * 4
    df = pl.DataFrame({"x": xs, "y": ys, "presence": [1, 0, 0, 0]})
    gdf, stats = m.hotspots(cfg, str(proba_path), df, seuil=0.5, n_hotspots=5)
    assert stats["n_hotspots"] >= 1
    top1 = gdf.iloc[0]
    assert top1["effort_ebird"] == 0                    # hotspot = cellule non surveillée
    assert top1["proba"] >= 0.5
    # la cellule (2,2) (centre ≈ left+2500, top-2500) est retenue ; la (0,0) surveillée est exclue
    assert abs(top1["x_centre"] - (left + 2500)) < 600
    assert (gdf["effort_ebird"] == 0).all()
