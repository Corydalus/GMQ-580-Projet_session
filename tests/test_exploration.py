"""Tests de code/06_exploration_elevation.py — résidualisation + construction des variantes.

Module à préfixe numérique : chargé via importlib. Fixtures synthétiques légères.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
import numpy as np  # noqa: E402


def _charger_expl():
    chemin = Path(__file__).resolve().parent.parent / "code" / "06_exploration_elevation.py"
    spec = importlib.util.spec_from_file_location("expl06", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_residualiser_retire_tendance_xy():
    e = _charger_expl()
    rng = np.random.default_rng(0)
    x, y = rng.random(200), rng.random(200)
    col = 3 * x - 2 * y + 5 + rng.normal(0, 0.01, 200)   # tendance planaire + bruit faible
    res = e.residualiser(col, x, y)
    assert abs(np.corrcoef(res, x)[0, 1]) < 0.05         # plus de corrélation linéaire avec x
    assert abs(np.corrcoef(res, y)[0, 1]) < 0.05         # ni avec y
    assert res.std() < col.std()                         # variance réduite (tendance retirée)


def test_variantes_habitat_formes():
    e = _charger_expl()
    m = e.charger_module_modele()
    rng = np.random.default_rng(1)
    n, k = 50, len(m.FEATURES_HABITAT)
    x_hab = rng.random((n, k))
    xc, yc = rng.random(n) * 1000, rng.random(n) * 1000
    v = e.variantes_habitat(m, x_hab, xc, yc)
    assert v["baseline"][0].shape == (n, k)
    assert v["sans_elevation"][0].shape == (n, k - 1)
    assert "elevation" not in v["sans_elevation"][1]
    assert v["elevation_residualisee"][0].shape == (n, k)
    assert "elevation_residuelle" in v["elevation_residualisee"][1]
    je = m.FEATURES_HABITAT.index("elevation")
    assert not np.allclose(v["elevation_residualisee"][0][:, je], x_hab[:, je])  # colonne modifiée
