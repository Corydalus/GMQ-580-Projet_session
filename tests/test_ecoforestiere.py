"""Tests de `code/ecoforestiere.py` — encodage MFFP + rasterisation (J3)."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
eco = pytest.importorskip("ecoforestiere")     # code/ ajouté au path par conftest


def test_age_ordinal():
    assert eco.age_ordinal(None) == 0                  # non-forêt
    assert eco.age_ordinal(float("nan")) == 0
    assert eco.age_ordinal("10") == 1                  # jeune
    assert eco.age_ordinal("JIN") == 1                 # jeune inéquienne
    assert eco.age_ordinal("50") == 2                  # mature
    assert eco.age_ordinal("70") == 3                  # âgé
    assert eco.age_ordinal("120") == 4                 # vieux
    assert eco.age_ordinal("VIN") == 4                 # vieux inéquienne
    assert eco.age_ordinal("7050") == 3                # composite → 1er terme (70)
    assert eco.age_ordinal("VIN30") == 4               # composite → VIN
    assert eco.age_ordinal("30VIN") == 2               # composite → 30
    assert eco.age_ordinal("XYZ") == 0                 # inconnu → non-forêt


def test_densite_pct():
    assert [eco.densite_pct(c) for c in ["A", "B", "C", "D"]] == [90, 70, 50, 32]
    assert eco.densite_pct(None) == 0 and eco.densite_pct(float("nan")) == 0


def test_couvert_code():
    assert [eco.couvert_code(c) for c in ["F", "M", "R"]] == [1, 2, 3]
    assert eco.couvert_code(None) == 0


def test_rasteriser_champ():
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame({"cl_age": ["50", "JIN"]},
                           geometry=[box(0, 0, 20, 40), box(20, 0, 40, 40)], crs="EPSG:32198")
    out = eco.rasteriser_champ(gdf, "cl_age", eco.age_ordinal,
                               from_origin(0, 40, 10, 10), 4, 4, fill=255)
    assert out.shape == (4, 4)
    assert (out[:, :2] == 2).all()                     # polygone "50" → mature (2)
    assert (out[:, 2:] == 1).all()                     # polygone "JIN" → jeune (1)
