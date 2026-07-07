"""Fixtures partagées de la suite de tests — fixtures synthétiques légères.

Principe : aucun accès disque hors `tmp_path`, aucun réseau, jamais les données
réelles (~150 Go). Les imports lourds (numpy…) sont paresseux (`importorskip`)
pour que la collecte pytest fonctionne même sans l'environnement complet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    """Racine du dépôt."""
    return PROJECT_ROOT


@pytest.fixture
def sample_config_dict() -> dict:
    """Config minimale valide — miroir du squelette de CLAUDE.md §3.3.

    Sert de base aux tests de validation (J1) : on la copie puis on introduit
    une valeur invalide pour vérifier que `code/config.py` la rejette.
    """
    return {
        "espece": {
            "nom_latin": "Antrostomus vociferus",
            "nom_commun": "Engoulevent bois-pourri",
            "code_ebird": "eawpwi",
        },
        "zone_etude": {
            "gpkg": "data/zone_etude.gpkg",
            "crs": "EPSG:32198",
            "resolution_m": 5,
        },
        "ebird": {
            "mois_saison": [6, 7],
            "duree_max_min": 300,
            "distance_max_km": 5,
            "observateurs_max": 10,
            "protocoles": ["Stationary", "Traveling"],
            "listes_completes": True,
            "buffer_m": 30,
        },
        "climat_stac": {
            "collection": "landsat-c2-l2",
            "fournisseur": "planetary-computer",
            "mois": [6, 7],
            "annees": None,
            "couverture_nuageuse_max": 60,
            "reducteur": "median",
        },
        "modele": {
            "random_state": 42,
            "bloc_cv_km": 10,
            "n_folds": 5,
            "n_iter_recherche": 20,
        },
        "chemins": {
            "raw": "data/raw",
            "interim": "data/interim",
            "processed": "data/processed",
            "outputs": "outputs",
        },
    }


@pytest.fixture
def synthetic_raster():
    """Petit tableau 5×5 (numpy) pour tester focal stats / MESS (J1, J6)."""
    np = pytest.importorskip("numpy")
    return np.arange(25, dtype="float32").reshape(5, 5)
