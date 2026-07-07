"""Test de fumée — valide que la CI s'exécute et que la structure du dépôt est intacte.

Sans dépendance lourde : ne requiert que `pytest` + la bibliothèque standard.
Les vrais tests de logique arrivent avec chaque jalon (voir CLAUDE.md §5 / §8).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_verite():
    """La CI tourne."""
    assert True


def test_scripts_pipeline_presents():
    """Les six scripts du pipeline existent."""
    code = ROOT / "code"
    attendus = [
        "utils.py",
        "01_predictors.py",
        "02_ebird.py",
        "03_model.py",
        "04_predict.py",
        "05_figures.py",
    ]
    manquants = [nom for nom in attendus if not (code / nom).exists()]
    assert not manquants, f"Scripts manquants : {manquants}"


def test_pyproject_present():
    """Le pyproject est présent (socle de reproductibilité)."""
    assert (ROOT / "pyproject.toml").is_file()


def test_config_dict_fixture(sample_config_dict):
    """La fixture de config expose bien les blocs attendus."""
    for bloc in ("espece", "zone_etude", "ebird", "climat_stac", "modele", "chemins"):
        assert bloc in sample_config_dict
