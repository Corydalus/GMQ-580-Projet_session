"""Tests de `code/02_ebird.py` — filtres d'effort, variables de détection, covariables (J4).

Le module a un préfixe numérique (non importable directement) : on le charge via importlib.
Fixtures synthétiques légères uniquement — aucun accès réseau ni aux données réelles.
"""

from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("polars")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def _charger_ebird():
    chemin = Path(__file__).resolve().parent.parent / "code" / "02_ebird.py"
    spec = importlib.util.spec_from_file_location("ebird02", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(sample_config_dict: dict):
    from config import Config
    d = {**sample_config_dict, "ebird": {**sample_config_dict["ebird"], "annee_min": 2010}}
    return Config.model_validate(d)


# ── Filtres d'effort + saison + année + emprise ──────────────────────────────

def _ligne(**kw):
    """Ligne de zero-fill (colonnes str, comme le CSV eBird) avec des défauts qui passent."""
    base = dict(observation_date="2020-06-15", protocol_name="Stationary",
                duration_minutes="60", effort_distance_km="1", number_observers="2",
                latitude="45.5", longitude="-72.0", all_species_reported="TRUE",
                species_observed="TRUE")
    base.update(kw)
    return base


def test_filtrer_checklists(sample_config_dict):
    m = _charger_ebird()
    bbox = (-73.0, 45.0, -71.0, 46.0)
    lignes = [
        _ligne(species_observed="TRUE"),                      # ✓ présence
        _ligne(effort_distance_km="NA", protocol_name="Stationary",
               species_observed="FALSE"),                     # ✓ absence, distance NA → 0
        _ligne(observation_date="2020-09-15"),                # ✗ hors saison
        _ligne(duration_minutes="400"),                       # ✗ durée > 300
        _ligne(effort_distance_km="10"),                      # ✗ distance > 5
        _ligne(number_observers="20"),                        # ✗ > 10 observateurs
        _ligne(all_species_reported="FALSE"),                 # ✗ liste incomplète
        _ligne(protocol_name="Incidental"),                   # ✗ protocole exclu
        _ligne(longitude="-80.0"),                            # ✗ hors emprise
        _ligne(observation_date="2005-06-15"),                # ✗ avant annee_min
    ]
    lf = pl.DataFrame(lignes).lazy()
    out = m.filtrer_checklists(lf, _cfg(sample_config_dict), bbox).collect()
    assert out.height == 2                                    # seules les 2 premières passent
    assert int(out["presence"].sum()) == 1                   # 1 présence, 1 absence
    assert set(out["presence"].to_list()) == {0, 1}


# ── Variables de détection ───────────────────────────────────────────────────

def test_jour_julien_et_log_duree():
    m = _charger_ebird()
    assert m.jour_julien(datetime.date(2020, 1, 1)) == 1
    assert m.jour_julien(datetime.date(2020, 12, 31)) == 366   # année bissextile
    assert m.jour_julien(datetime.date(2021, 12, 31)) == 365
    assert m.log_duree(0) == 0.0
    assert m.log_duree(float(np.expm1(1))) == pytest.approx(1.0)


def test_phase_lune_bornes():
    m = _charger_ebird()
    p = m.phase_lune(datetime.date(2024, 6, 21))
    assert 0.0 <= p < 28.0


def test_minutes_apres_coucher_signe():
    m = _charger_ebird()
    lat, lon, date, tz = 45.5, -72.0, datetime.date(2020, 6, 15), "America/Toronto"
    soir = m.minutes_apres_coucher(lat, lon, date, datetime.time(22, 30), tz)
    apres_minuit = m.minutes_apres_coucher(lat, lon, date, datetime.time(0, 30), tz)
    aprem = m.minutes_apres_coucher(lat, lon, date, datetime.time(14, 0), tz)
    assert soir is not None and 0 < soir < 720               # ~2 h après le coucher (~20 h 40)
    assert apres_minuit is not None and apres_minuit > 0     # continuité de la nuit après minuit
    assert aprem is not None and aprem < 0                   # après-midi = avant le coucher
    assert all(-720 <= v < 720 for v in (soir, apres_minuit, aprem))  # repli cyclique
    assert m.minutes_apres_coucher(lat, lon, date, None, tz) is None  # heure absente


def test_parse_heure():
    m = _charger_ebird()
    assert m._parse_heure("22:05:00") == datetime.time(22, 5, 0)
    assert m._parse_heure("07:10") == datetime.time(7, 10)
    assert m._parse_heure("NA") is None
    assert m._parse_heure(None) is None


# ── Extraction des covariables (buffer sur mini-stack) ───────────────────────

def _mini_stack(path: Path, valeurs=(10.0, 20.0)):
    """Écrit un mini-stack (2 bandes constantes) EPSG:32198, 20×20 px à 5 m."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    tr = from_origin(-300000, 200000, 5, 5)
    prof = dict(driver="GTiff", width=20, height=20, count=len(valeurs), dtype="float32",
                crs="EPSG:32198", transform=tr, nodata=float("nan"))
    with rasterio.open(path, "w", **prof) as ds:
        for i, v in enumerate(valeurs, 1):
            ds.write(np.full((20, 20), v, dtype="float32"), i)
            ds.set_band_description(i, f"bande{i}")
    return tr


def test_extraire_covariables_buffer(tmp_path):
    m = _charger_ebird()
    path = tmp_path / "mini_stack.tif"
    _mini_stack(path, valeurs=(10.0, 20.0))
    # point au centre (dans l'emprise) + point loin à l'ouest (hors emprise)
    xs = np.array([-299950.0, -320000.0])
    ys = np.array([199950.0, 199950.0])
    vals, noms = m.extraire_covariables_buffer(path, xs, ys, buffer_m=30.0)
    assert noms == ["bande1", "bande2"]
    assert vals.shape == (2, 2)
    assert vals[0, 0] == pytest.approx(10.0)                  # moyenne buffer = constante
    assert vals[0, 1] == pytest.approx(20.0)
    assert np.isnan(vals[1]).all()                            # hors emprise → NaN


def test_extraire_gere_les_nan_partiels(tmp_path):
    """Un buffer chevauchant du nodata renvoie la moyenne des seuls pixels valides."""
    m = _charger_ebird()
    rasterio = pytest.importorskip("rasterio")
    path = tmp_path / "stack_nan.tif"
    _mini_stack(path, valeurs=(10.0, 20.0))
    with rasterio.open(path, "r+") as ds:
        b = ds.read(1)
        b[:, :10] = np.nan                                   # moitié gauche = nodata
        ds.write(b, 1)
    xs, ys = np.array([-299950.0]), np.array([199950.0])     # centre, buffer chevauche la coupure
    vals, _ = m.extraire_covariables_buffer(path, xs, ys, buffer_m=30.0)
    assert vals[0, 0] == pytest.approx(10.0)                  # pixels valides seulement
    assert vals[0, 1] == pytest.approx(20.0)
