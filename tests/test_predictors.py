"""Tests de `code/01_predictors.py` — spec du stack 10 bandes (J3).

Le module a un préfixe numérique (non importable directement) : on le charge via importlib.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")


def _charger_predictors():
    chemin = Path(__file__).resolve().parent.parent / "code" / "01_predictors.py"
    spec = importlib.util.spec_from_file_location("predictors01", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stack_bandes_ordre_et_noms():
    m = _charger_predictors()
    noms = [desc for _, desc in m.BANDES_STACK]
    assert len(m.BANDES_STACK) == 10                         # 10 variables (§2)
    assert len(set(noms)) == 10                              # noms uniques
    # ordre canonique : quelques positions clés
    assert noms[0] == "MHC_hauteur_canopee"                  # var 1
    assert noms[5] == "LST_estivale"                         # var 6
    assert noms[8] == "distance_milieu_humide"               # var 9
    assert noms[9] == "elevation"                            # var 10
    # chaque bande pointe vers un fichier interim distinct
    fichiers = [f for f, _ in m.BANDES_STACK]
    assert len(set(fichiers)) == 10
