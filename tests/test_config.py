"""Tests de `code/config.py` — validation typée de la configuration (J1)."""

from __future__ import annotations

import copy

import pytest

config = pytest.importorskip("config")          # code/ ajouté au path par conftest
from pydantic import ValidationError            # noqa: E402


def test_config_valide(sample_config_dict):
    cfg = config.Config.model_validate(sample_config_dict)
    assert cfg.espece.nom_latin == "Antrostomus vociferus"
    assert cfg.zone_etude.crs == "EPSG:32198"
    assert cfg.ebird.mois_saison == [6, 7]
    assert cfg.modele.random_state == 42
    # Blocs à défaut absents du dict → valeurs par défaut appliquées
    assert cfg.calcul.threads_per_worker == 2
    assert cfg.variables.routes_par_type is False


def test_config_rejette_mois_invalide(sample_config_dict):
    bad = copy.deepcopy(sample_config_dict)
    bad["ebird"]["mois_saison"] = [13]
    with pytest.raises(ValidationError):
        config.Config.model_validate(bad)


def test_config_rejette_reducteur_invalide(sample_config_dict):
    bad = copy.deepcopy(sample_config_dict)
    bad["climat_stac"]["reducteur"] = "moyenne"
    with pytest.raises(ValidationError):
        config.Config.model_validate(bad)


def test_config_rejette_fenetre_detection_invalide(sample_config_dict):
    bad = copy.deepcopy(sample_config_dict)
    bad["figures"] = {"fenetre_detection_apres_coucher_min": [450, -60]}  # début ≥ fin
    with pytest.raises(ValidationError):
        config.Config.model_validate(bad)


def test_config_rejette_hotspots_quantile_invalide(sample_config_dict):
    bad = copy.deepcopy(sample_config_dict)
    bad["hotspots"] = {"effort_quantile": 1.5}   # hors [0, 1]
    with pytest.raises(ValidationError):
        config.Config.model_validate(bad)


def test_config_rejette_cle_inconnue(sample_config_dict):
    bad = copy.deepcopy(sample_config_dict)
    bad["zone_etude"]["crss"] = "EPSG:4326"      # faute de frappe → extra="forbid"
    with pytest.raises(ValidationError):
        config.Config.model_validate(bad)


def test_config_rejette_crs_non_epsg(sample_config_dict):
    bad = copy.deepcopy(sample_config_dict)
    bad["zone_etude"]["crs"] = "Quebec Lambert"
    with pytest.raises(ValidationError):
        config.Config.model_validate(bad)


def test_load_config_yaml(tmp_path, sample_config_dict):
    import yaml
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(sample_config_dict), encoding="utf-8")
    cfg = config.load_config(p)
    assert cfg.espece.code_ebird == "eawpwi"


def test_load_config_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load_config(tmp_path / "inexistant.yaml")


def test_config_yaml_du_depot_est_valide():
    """Le config.yaml versionné à la racine doit valider."""
    from pathlib import Path
    racine = Path(__file__).resolve().parents[1]
    cfg = config.load_config(racine / "config.yaml")
    assert cfg.zone_etude.resolution_m == 5
