"""07 — Comparaison des cartes de potentiel : avec vs sans élévation (exploration).

Juxtapose la carte de probabilité de production (avec élévation) et celle de la variante
sans élévation (`config_sans_elevation.yaml`), à la même échelle, pour visualiser l'effet du
retrait de l'élévation sur la distribution spatiale du potentiel d'habitat.
Réutilise `fig_comparaison_cartes` de 05_figures.

Prérequis : 04_predict.py exécuté pour chaque config (cartes proba_5m.tif présentes).
Sortie : outputs/figures/comparaison_elevation_proba.png
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import geopandas as gpd

import utils
from config import load_config

log = logging.getLogger("07_comparaison")


def _module_figures():
    """Charge code/05_figures.py (préfixe numérique) pour réutiliser fig_comparaison_cartes."""
    chemin = Path(__file__).resolve().parent / "05_figures.py"
    spec = importlib.util.spec_from_file_location("figures05", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--config-variante", default="config_sans_elevation.yaml")
    args, _ = parser.parse_known_args()

    base = load_config(args.config)
    var = load_config(args.config_variante)
    utils.setup_logging("07_comparaison", log_dir=f"{base.chemins.comparaison}/logs")
    fig = _module_figures()

    zone = gpd.read_file(base.zone_etude.gpkg).to_crs(base.zone_etude.crs)
    routes_path = Path(base.chemins.interim) / "routes_zone.gpkg"
    routes = gpd.read_file(routes_path).to_crs(base.zone_etude.crs) if routes_path.exists() else None
    if routes is not None and "ClsRte" in routes.columns:
        routes = routes[routes["ClsRte"].isin(base.figures.routes_classes_principales)]

    proba_base = f"{base.chemins.outputs}/maps/proba_5m.tif"
    proba_var = f"{var.chemins.outputs}/maps/proba_5m.tif"
    for p in (proba_base, proba_var):
        if not Path(p).exists():
            raise FileNotFoundError(f"Carte manquante : {p} (lancer 04_predict avec la config voulue).")

    fig_dir = Path(base.chemins.comparaison) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sortie = str(fig_dir / "comparaison_elevation_proba.png")
    with utils.log_step("Comparaison proba avec/sans élévation", log):
        fig.fig_comparaison_cartes(
            [proba_base, proba_var],
            ["Avec élévation (production)", "Sans élévation (habitat)"],
            sortie, zone=zone, routes=routes)
    log.info("  figure → %s", sortie)


if __name__ == "__main__":
    main()
