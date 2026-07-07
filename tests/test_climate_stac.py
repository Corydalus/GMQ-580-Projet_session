"""Tests de `code/climate_stac.py` — logique du composite LST (J2).

Fonctions pures testées sur cubes synthétiques (aucun réseau). Un test d'intégration
STAC réel est marqué `network` (exclu de la CI).
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
import numpy as np                              # noqa: E402

cs = pytest.importorskip("climate_stac")        # code/ ajouté au path par conftest


def _dn(celsius: float) -> float:
    """DN brut `ST_B10` correspondant à une température °C (inverse de la conversion)."""
    return (celsius + cs.KELVIN - cs.ST_OFFSET) / cs.ST_SCALE


def test_conversion_celsius_plausible():
    dn = np.array([_dn(0.0), _dn(20.0), _dn(35.0)])
    c = cs.st_b10_en_celsius(dn)
    assert np.allclose(c, [0.0, 20.0, 35.0], atol=0.01)


def test_conversion_fill_est_nan():
    assert np.isnan(cs.st_b10_en_celsius(np.array([0.0])))[0]


def test_masque_qa():
    # 0 = fill/remplissage-odc (rejeté) ; bit 3 (cloud=8), bit 4 (shadow=16), bit 0 (fill=1)
    # = rejet ; bit 6 (clear=64) seul = OK.
    qa = np.array([0, 8, 16, 1, 64], dtype="uint16")
    assert list(cs.masque_clair_qa(qa)) == [False, False, False, False, True]


def test_composite_median_ignore_nuages():
    # pixel A : 3 scènes claires (10, 20, 30 °C) → médiane 20.
    # pixel B : 1 claire (25 °C) + 2 nuageuses → médiane = 25.
    dn = np.array([
        [[_dn(10.0), _dn(25.0)]],
        [[_dn(20.0), _dn(99.0)]],
        [[_dn(30.0), _dn(99.0)]],
    ])
    qa = np.array([
        [[64, 64]],   # 64 = clair (bit 6)
        [[64, 8]],    # B nuageux
        [[64, 8]],    # B nuageux
    ], dtype="uint16")
    lst, n_clair = cs.composite_lst(dn, qa, reducteur="median")
    assert lst[0, 0] == pytest.approx(20.0, abs=0.01)
    assert lst[0, 1] == pytest.approx(25.0, abs=0.01)
    assert n_clair[0, 0] == 3 and n_clair[0, 1] == 1


def test_composite_pixel_tout_nuageux_est_nan():
    dn = np.array([[[_dn(20.0)]], [[_dn(21.0)]]])
    qa = np.array([[[8]], [[8]]], dtype="uint16")   # toujours nuageux
    lst, n_clair = cs.composite_lst(dn, qa)
    assert np.isnan(lst[0, 0]) and n_clair[0, 0] == 0


def test_composite_scene_qa_remplie_zero_exclue():
    # Asset qa illisible → odc remplit à 0. Le dn reste réel : la scène doit être EXCLUE
    # (qa==0 = fill), sinon elle fuite non masquée dans la médiane.
    dn = np.array([[[_dn(20.0)]], [[_dn(99.0)]]])   # t0 clair 20°C ; t1 dn réel mais qa=0
    qa = np.array([[[64]], [[0]]], dtype="uint16")
    lst, n_clair = cs.composite_lst(dn, qa)
    assert lst[0, 0] == pytest.approx(20.0, abs=0.01)   # 99°C exclu → pas de fuite
    assert n_clair[0, 0] == 1


def test_intervalles_datetime():
    inter = cs.intervalles_datetime([2020, 2021], [6, 7])
    assert inter == ["2020-06-01/2020-07-31", "2021-06-01/2021-07-31"]


# ── Checkpoint / reprise par tuile ──────────────────────────────────────────

def test_bbox_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    assert cs.bbox_overlap(a, (5.0, 5.0, 15.0, 15.0))       # chevauchement
    assert cs.bbox_overlap(a, (-5.0, -5.0, 1.0, 1.0))       # coin
    assert not cs.bbox_overlap(a, (11.0, 0.0, 20.0, 10.0))  # à droite, disjoint
    assert not cs.bbox_overlap(a, (0.0, 11.0, 10.0, 20.0))  # au-dessus, disjoint


def test_tuiles_a_traiter_reprise(tmp_path):
    # Grille 3×3, tuiles 2×2 → 4 tuiles aux coins (0,0),(0,2),(2,0),(2,2).
    # On pré-crée le checkpoint (0,0) : il doit être marqué « déjà fait » (repris).
    cs.chemin_tuile(tmp_path, 0, 0).touch()
    taches = list(cs.tuiles_a_traiter(3, 3, 2, tmp_path))
    faits = {(r, c): deja for r, c, w, h, p, deja in taches}
    assert set(faits) == {(0, 0), (0, 2), (2, 0), (2, 2)}
    assert faits[(0, 0)] is True
    assert not any(faits[k] for k in [(0, 2), (2, 0), (2, 2)])


def test_ecrire_tuile_atomique_et_mosaique(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    from rasterio.transform import from_origin

    res, x0, y0 = 30.0, 250_000.0, 5_010_000.0
    gauche = np.array([[1, 2], [3, 4]], dtype="float32")
    droite = np.array([[5, 6], [7, 8]], dtype="float32")
    nclair = np.full((2, 2), 3, dtype="float32")
    pg = cs.ecrire_tuile(tmp_path / "lst_r00000_c00000.tif", gauche, nclair,
                         from_origin(x0, y0, res, res), "EPSG:32198")
    pd = cs.ecrire_tuile(tmp_path / "lst_r00000_c00002.tif", droite, nclair,
                         from_origin(x0 + 2 * res, y0, res, res), "EPSG:32198")

    # écriture atomique : pas de fichier .tmp résiduel, checkpoint bien présent
    assert pg.exists() and not (tmp_path / "lst_r00000_c00000.tif.tmp").exists()
    with rasterio.open(pg) as ds:
        assert ds.count == 2                              # LST + n_clair

    mosaic = cs.mosaiquer_tuiles([pg, pd])
    lst = mosaic.isel(band=0).values
    assert lst.shape == (2, 4)
    assert np.array_equal(lst, [[1, 2, 5, 6], [3, 4, 7, 8]])


@pytest.mark.network
def test_stac_recherche_reelle():
    """Intégration : la requête STAC Planetary Computer renvoie des scènes (réseau requis)."""
    bbox = (-72.75, 45.25, -71.5, 46.125)          # emprise zone d'étude (EPSG:4326)
    items = cs.rechercher_items(bbox, [2022], [6, 7], cloud_max=40)
    assert len(items) > 0
    assert "lwir11" in items[0].assets
