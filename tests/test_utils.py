"""Tests de `code/utils.py` — focal stats, MESS, COG, rapport (J1).

Fixtures synthétiques uniquement ; aucun accès réseau ni donnée réelle.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
import numpy as np                              # noqa: E402

utils = pytest.importorskip("utils")            # code/ ajouté au path par conftest


def test_focal_mean_constant():
    a = np.full((7, 7), 3.0)
    out = utils.focal_mean(a, size=3)
    assert np.allclose(out, 3.0)


def test_focal_mean_ignore_nan():
    a = np.ones((3, 3))
    a[1, 1] = np.nan
    out = utils.focal_mean(a, size=3)
    # La cellule centrale = moyenne des 8 voisins valides (= 1.0), pas NaN.
    assert np.isfinite(out[1, 1])
    assert out[1, 1] == pytest.approx(1.0)


def test_focal_sum_compte():
    a = np.zeros((5, 5))
    a[2, 2] = 1.0
    out = utils.focal_sum(a, size=3)
    # Toute fenêtre 3x3 contenant le pixel central voit une somme de 1.
    assert out[2, 2] == pytest.approx(1.0)
    assert out[0, 0] == pytest.approx(0.0)


def test_focal_proportion():
    mask = np.zeros((5, 5))
    mask[:, :] = 1.0
    out = utils.focal_proportion(mask, size=3)
    assert out[2, 2] == pytest.approx(1.0)


def test_mess_interieur_positif_exterieur_negatif():
    ref = [np.arange(0.0, 11.0)]          # référence 0..10 pour une variable
    proj = np.array([[[5.0, -5.0, 15.0]]])  # (1 var, 1, 3) : dedans, sous min, au-dessus max
    m = utils.mess(proj, ref)
    assert m[0, 0] > 0        # 5 est au centre de la plage → similarité positive
    assert m[0, 1] < 0        # -5 sous le minimum → extrapolation (négatif)
    assert m[0, 2] < 0        # 15 au-dessus du maximum → extrapolation (négatif)


def test_mess_min_sur_variables():
    # 2 variables : la 2e extrapole → le MESS (minimum) doit être négatif.
    ref = [np.arange(0.0, 11.0), np.arange(0.0, 11.0)]
    proj = np.array([[[5.0]], [[20.0]]])  # (2 var, 1, 1)
    m = utils.mess(proj, ref)
    assert m[0, 0] < 0


def test_ecrire_rapport_json(tmp_path):
    p = utils.ecrire_rapport_json("essai", {"n_scenes": 12, "pct_manquant": 0.3}, log_dir=tmp_path)
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["script"] == "essai" and d["n_scenes"] == 12
    assert "horodatage" in d


def test_write_cog_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    arr = np.arange(64, dtype="float32").reshape(8, 8)
    transform = from_origin(0, 8, 1, 1)
    p = utils.write_cog(tmp_path / "t.tif", arr, transform, crs="EPSG:32198")
    with rasterio.open(p) as ds:
        assert ds.count == 1
        assert ds.crs.to_epsg() == 32198
        assert np.allclose(ds.read(1), arr)


def test_iter_windows_couvre_tout():
    fenetres = list(utils.iter_windows(10, 10, tile=4))
    aire = sum(w.width * w.height for w in fenetres)
    assert aire == 100          # 10x10 entièrement couvert, sans chevauchement
