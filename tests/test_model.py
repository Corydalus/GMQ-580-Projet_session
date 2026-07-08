"""Tests de `code/03_model.py` — blocs spatiaux, CV, TSS, pipeline RF (J5).

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


def _charger_model():
    chemin = Path(__file__).resolve().parent.parent / "code" / "03_model.py"
    spec = importlib.util.spec_from_file_location("model03", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(sample_config_dict: dict, **modele):
    from config import Config
    d = {**sample_config_dict, "modele": {**sample_config_dict["modele"], **modele}}
    return Config.model_validate(d)


# ── Blocs spatiaux 10 km ─────────────────────────────────────────────────────

def test_assigner_blocs_10km():
    m = _charger_model()
    x = np.array([0.0, 2000.0, 15000.0, 0.0])
    y = np.array([0.0, 0.0, 0.0, 15000.0])
    b = m.assigner_blocs(x, y, 10000.0)
    assert b[0] == b[1]          # 0 et 2 km → même bloc de 10 km
    assert b[0] != b[2]          # 15 km en x → bloc différent
    assert b[0] != b[3]          # 15 km en y → bloc différent
    assert b[2] != b[3]          # blocs distincts entre eux


# ── TSS ──────────────────────────────────────────────────────────────────────

def test_tss_optimal():
    m = _charger_model()
    tss, seuil = m.tss_optimal(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert tss == pytest.approx(1.0)                 # parfaitement séparable
    assert 0.2 < seuil <= 0.8
    tss0, _ = m.tss_optimal(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert tss0 == pytest.approx(0.0, abs=1e-9)      # scores constants → TSS nul


# ── CV spatiale : aucun bloc à cheval train/test ────────────────────────────

@pytest.mark.parametrize("stratifie", [True, False])
def test_cv_separe_les_blocs(sample_config_dict, stratifie):
    m = _charger_model()
    rng = np.random.default_rng(0)
    n = 150
    groups = rng.integers(0, 20, n)                  # 20 blocs
    y = rng.integers(0, 2, n)
    X = rng.random((n, len(m.FEATURES)))
    cv = m.make_cv(_cfg(sample_config_dict, n_folds=3, cv_stratifie=stratifie))
    n_splits = 0
    for itr, ite in cv.split(X, y, groups):
        assert set(groups[itr]).isdisjoint(set(groups[ite]))   # pas de fuite de bloc
        n_splits += 1
    assert n_splits == 3


# ── Pipeline : le RF s'entraîne et évalue sur un mini-jeu synthétique ────────

def test_pipeline_entraine(sample_config_dict):
    m = _charger_model()
    rng = np.random.default_rng(1)
    # 12 blocs (6 « présence », 6 « absence »), signal porté par la variable 0
    blocs, X_parts, y_parts, g_parts = 12, [], [], []
    for g in range(blocs):
        classe = g % 2
        Xg = rng.random((12, len(m.FEATURES)))
        Xg[:, 0] += classe                            # variable 0 sépare les classes
        X_parts.append(Xg)
        y_parts.append(np.full(12, classe))
        g_parts.append(np.full(12, g))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)
    cfg = _cfg(sample_config_dict, n_folds=3, cv_stratifie=True)
    cv = m.make_cv(cfg)
    metr = m.evaluer_spatial({"n_estimators": 20, "max_depth": 5}, X, y, groups, cv, cfg)
    assert set(metr) >= {"auc_moy", "auc_std", "tss_moy", "tss_std", "auc_folds", "seuils"}
    assert len(metr["auc_folds"]) == 3
    assert 0.0 <= metr["auc_moy"] <= 1.0
    assert metr["auc_moy"] > 0.8                      # signal net → bonne séparation
