"""05 — Figures : toutes les figures du rapport (matplotlib + contextily).

Génère, à partir des sorties J5/J6, les PNG du rapport :
- cartes probabilité / incertitude (variance inter-arbres) / MESS — **raster opaque** avec
  le réseau routier du Québec en surimpression, l'emprise de la zone et les hotspots ;
- une **carte de localisation** (fond OSM contextily) pour situer la zone d'étude ;
- importance par permutation (modèle habitat), PDP des 10 variables d'habitat,
  métriques de validation croisée spatiale (AUC-ROC + TSS par fold, 2 modèles).

Sorties : outputs/figures/*.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless (aucun affichage requis)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import utils  # noqa: E402
from config import load_config_from_cli  # noqa: E402

log = logging.getLogger("05_figures")

ETIQUETTES = {
    "MHC_hauteur_canopee": "Hauteur de canopée (m)", "TWI": "TWI",
    "densite_lisiere": "Densité de lisière", "prop_feuillu_melange": "Prop. feuillu/mélangé",
    "densite_routes": "Densité de routes (km/km²)", "LST_estivale": "LST estivale (°C)",
    "classe_age": "Classe d'âge", "densite_peuplement": "Densité de peuplement (%)",
    "distance_milieu_humide": "Distance milieu humide (m)", "elevation": "Élévation (m)",
}


def _lire_decime(path: str, max_dim: int = 1600, fidele: bool = False):
    """Lit un raster décimé pour l'affichage. Retourne (array, extent, crs).

    `fidele=True` : sous-échantillonne des pixels réels (bandes de lignes plein-réso) plutôt
    que les overviews moyennés — indispensable pour le MESS, dont de rares valeurs extrêmes
    faussent les moyennes de blocs.
    """
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as ds:
        fac = max(1, int(max(ds.width, ds.height) / max_dim))
        b = ds.bounds
        extent = (b.left, b.right, b.bottom, b.top)
        if not fidele:
            out = (ds.height // fac, ds.width // fac)
            return ds.read(1, out_shape=out, masked=True).filled(np.nan), extent, ds.crs
        parts = []
        for r in range(0, ds.height, 2048):
            h = min(2048, ds.height - r)
            bloc = ds.read(1, window=Window(0, r, ds.width, h), masked=True).filled(np.nan)
            parts.append(bloc[::fac, ::fac])
        return np.vstack(parts), extent, ds.crs


def _fond_osm(ax, crs, log_warn=True) -> None:
    """Ajoute un fond OSM (contextily) ; ignoré proprement sans réseau."""
    try:
        import contextily as cx
        cx.add_basemap(ax, crs=crs, source=cx.providers.OpenStreetMap.Mapnik, attribution_size=5)
    except Exception as e:  # noqa: BLE001 — réseau/tuiles indisponibles = non bloquant
        if log_warn:
            log.warning("Fond OSM ignoré (%s)", e)


def fig_carte(raster_path: str, titre: str, label: str, sortie: str, cmap: str = "viridis",
              vmin=None, vmax=None, centre0=False, fidele=False, zone=None, routes=None,
              hotspots=None) -> str:
    """Carte d'un raster de résultat : raster OPAQUE + routes + emprise + hotspots (sans fond OSM)."""
    arr, extent, _ = _lire_decime(raster_path, fidele=fidele)
    if centre0:
        lim = float(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 98))  # borne robuste (outliers)
        vmin, vmax, cmap = -lim, lim, "RdBu_r"
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", zorder=1)
    if routes is not None:
        routes.plot(ax=ax, color="0.15", linewidth=0.25, alpha=0.5, zorder=2)
    if zone is not None:
        zone.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=3)
    if hotspots is not None and len(hotspots):
        hotspots.boundary.plot(ax=ax, edgecolor="#00ff00", linewidth=1.5, zorder=4)
        for _, h in hotspots.iterrows():
            c = h.geometry.centroid
            ax.annotate(str(h["rang"]), (c.x, c.y), color="#00ff00", fontsize=9, ha="center",
                        va="center", fontweight="bold", zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(titre)
    ax.set_xlabel("X (m, EPSG:32198)")
    ax.set_ylabel("Y (m)")
    fig.colorbar(im, ax=ax, shrink=0.7, label=label)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return sortie


def fig_localisation(zone, sortie: str, hotspots=None, marge: float = 0.15, fond=True) -> str:
    """Carte de localisation : emprise de la zone (+ hotspots) sur fond OSM."""
    fig, ax = plt.subplots(figsize=(8, 8))
    zone.boundary.plot(ax=ax, edgecolor="red", linewidth=1.8, zorder=3)
    if hotspots is not None and len(hotspots):
        hotspots.centroid.plot(ax=ax, color="red", markersize=25, zorder=4)
    minx, miny, maxx, maxy = zone.total_bounds
    dx, dy = (maxx - minx) * marge, (maxy - miny) * marge
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)
    if fond:
        _fond_osm(ax, zone.crs)
    ax.set_title("Localisation de la zone d'étude")
    ax.set_xlabel("X (m, EPSG:32198)")
    ax.set_ylabel("Y (m)")
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return sortie


def fig_importance(importance: list[dict], titre: str, sortie: str) -> str:
    """Barres horizontales de l'importance par permutation (± écart-type), triées."""
    imp = sorted(importance, key=lambda d: d["importance"])
    noms = [ETIQUETTES.get(d["variable"], d["variable"]) for d in imp]
    val = [d["importance"] for d in imp]
    err = [d.get("ecart_type", 0) for d in imp]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(noms, val, xerr=err, color="#2c7fb8", ecolor="gray", capsize=3)
    ax.set_xlabel("Chute d'AUC-ROC (importance par permutation)")
    ax.set_title(titre)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return sortie


def fig_pdp(pdp: dict, sortie: str) -> str:
    """Grille des courbes de dépendance partielle des variables d'habitat."""
    noms = [k for k in pdp if k not in ("script", "horodatage")]
    n = len(noms)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, nom in zip(axes, noms):
        ax.plot(pdp[nom]["grille"], pdp[nom]["pd"], color="#238b45")
        ax.set_title(ETIQUETTES.get(nom, nom), fontsize=9)
        ax.set_ylabel("proba partielle", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Dépendance partielle — variables d'habitat (modèle habitat)")
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return sortie


def fig_metriques_cv(rapport: dict, sortie: str) -> str:
    """AUC-ROC et TSS par fold (points) + moyenne±écart-type, pour les 2 modèles."""
    modeles = [("combiné", rapport["modele_combine"]), ("habitat", rapport["modele_habitat"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cle, titre in zip(axes, ("auc", "tss"), ("AUC-ROC", "TSS")):
        for i, (_, m) in enumerate(modeles):
            folds = m[f"{cle}_folds"]
            ax.scatter([i] * len(folds), folds, color="gray", zorder=3, s=25)
            ax.errorbar(i, m[f"{cle}_moy"], yerr=m[f"{cle}_std"], fmt="o", color="#d95f0e",
                        capsize=5, markersize=9, zorder=4)
        ax.set_xticks(range(len(modeles)))
        ax.set_xticklabels([n for n, _ in modeles])
        ax.set_title(f"{titre} (CV spatiale 10 km)")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return sortie


def main() -> None:
    import geopandas as gpd

    cfg = load_config_from_cli()
    utils.setup_logging("05_figures", log_dir=f"{cfg.chemins.outputs}/logs")
    out = cfg.chemins.outputs
    fig_dir = Path(out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    maps = Path(out) / "maps"
    rapport_model = json.load(open(f"{out}/logs/03_model_rapport.json", encoding="utf-8"))
    pdp = json.load(open(f"{out}/models/pdp_habitat_rapport.json", encoding="utf-8"))
    zone = gpd.read_file(cfg.zone_etude.gpkg).to_crs(cfg.zone_etude.crs)
    routes_path = Path(cfg.chemins.interim) / "routes_zone.gpkg"
    routes = gpd.read_file(routes_path).to_crs(cfg.zone_etude.crs) if routes_path.exists() else None
    hs_path = Path(out) / "tables" / "hotspots.gpkg"
    hotspots = gpd.read_file(hs_path) if hs_path.exists() else None
    produits: dict = {}

    with utils.log_step("Carte de localisation (fond OSM)", log):
        produits["localisation"] = fig_localisation(zone, str(fig_dir / "localisation.png"), hotspots)

    with utils.log_step("Cartes de résultat (proba, incertitude, MESS + routes)", log):
        produits["carte_proba"] = fig_carte(
            str(maps / "proba_5m.tif"), "Probabilité de présence (crépuscule)", "probabilité",
            str(fig_dir / "carte_proba.png"), cmap="magma", vmin=0, vmax=1,
            zone=zone, routes=routes, hotspots=hotspots)
        produits["carte_incertitude"] = fig_carte(
            str(maps / "incertitude_5m.tif"), "Incertitude (variance inter-arbres)", "variance",
            str(fig_dir / "carte_incertitude.png"), cmap="cividis", vmin=0, zone=zone, routes=routes)
        produits["carte_mess"] = fig_carte(
            str(maps / "mess_5m.tif"), "MESS — zones d'extrapolation (négatif)", "MESS",
            str(fig_dir / "carte_mess.png"), centre0=True, fidele=True, zone=zone, routes=routes)

    with utils.log_step("Importance, PDP, métriques CV", log):
        produits["importance"] = fig_importance(
            rapport_model["modele_habitat"]["importance"],
            "Importance par permutation — variables d'habitat",
            str(fig_dir / "importance_habitat.png"))
        produits["pdp"] = fig_pdp(pdp, str(fig_dir / "pdp_habitat.png"))
        produits["metriques_cv"] = fig_metriques_cv(rapport_model, str(fig_dir / "metriques_cv.png"))

    for nom, chemin in produits.items():
        log.info("  %s → %s", nom, chemin)
    utils.ecrire_rapport_json("05_figures", {"figures": produits}, log_dir=f"{out}/logs")


if __name__ == "__main__":
    main()
