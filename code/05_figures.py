"""05 — Figures : toutes les figures du rapport (matplotlib + contextily).

Génère, à partir des sorties J5/J6, les PNG du rapport :
- cartes probabilité / incertitude (variance inter-arbres) / MESS — **raster opaque** avec
  les **routes principales** (liseré blanc), l'emprise de la zone, les hotspots et une **légende** ;
- une **carte de localisation** (fond OSM contextily) pour situer la zone d'étude (sans hotspots) ;
- importance par permutation (modèle habitat), PDP des 10 variables d'habitat,
  métriques de validation croisée spatiale (AUC-ROC + TSS par fold, 2 modèles) ;
- **SHAP** (beeswarm + dépendance de l'élévation) et **diagnostic** du proxy spatial de l'élévation.

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
    """Carte d'un raster : raster OPAQUE + routes principales + emprise + hotspots + légende."""
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    arr, extent, _ = _lire_decime(raster_path, fidele=fidele)
    if centre0:
        lim = float(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 98))  # borne robuste (outliers)
        vmin, vmax, cmap = -lim, lim, "RdBu_r"
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", zorder=1)
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]  # filet blanc : routes visibles sur tout fond
    halo_zone = [pe.withStroke(linewidth=3.4, foreground="white")]
    handles = []
    if routes is not None and len(routes):
        routes.plot(ax=ax, color="black", linewidth=1.4, alpha=1.0, zorder=2, path_effects=halo)
        handles.append(Line2D([0], [0], color="black", lw=1.6, path_effects=halo,
                              label="Routes principales"))
    if zone is not None:
        zone.boundary.plot(ax=ax, edgecolor="black", linewidth=2.4, zorder=3, path_effects=halo_zone)
        handles.append(Line2D([0], [0], color="black", lw=2.4, path_effects=halo_zone,
                              label="Zone d'étude"))
    if hotspots is not None and len(hotspots):
        hotspots.boundary.plot(ax=ax, edgecolor="#00ff00", linewidth=1.5, zorder=4)
        for _, h in hotspots.iterrows():
            c = h.geometry.centroid
            ax.annotate(str(h["rang"]), (c.x, c.y), color="#00ff00", fontsize=9, ha="center",
                        va="center", fontweight="bold", zorder=5)
        handles.append(Line2D([0], [0], color="#00ff00", lw=1.5, label="Hotspots (rang)"))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
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


def fig_comparaison_cartes(rasters: list[str], titres: list[str], sortie: str, cmap: str = "magma",
                           vmin=0, vmax=1, label: str = "probabilité", zone=None, routes=None,
                           fidele=False) -> str:
    """Plusieurs rasters côte à côte, même échelle + colorbar partagée (comparaison de scénarios)."""
    import matplotlib.patheffects as pe
    n = len(rasters)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7.5), squeeze=False)
    axes = axes.ravel()
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    im = None
    for ax, rp, tt in zip(axes, rasters, titres):
        arr, extent, _ = _lire_decime(rp, fidele=fidele)
        im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest", zorder=1)
        if routes is not None and len(routes):
            routes.plot(ax=ax, color="black", linewidth=0.6, alpha=0.9, zorder=2, path_effects=halo)
        if zone is not None:
            zone.boundary.plot(ax=ax, edgecolor="black", linewidth=1.6, zorder=3, path_effects=halo)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(tt, fontsize=11)
        ax.set_xlabel("X (m, EPSG:32198)")
    axes[0].set_ylabel("Y (m)")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label=label)
    fig.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return sortie


def fig_localisation(zone, sortie: str, marge: float = 0.15, fond=True) -> str:
    """Carte de localisation : emprise de la zone d'étude sur fond OSM (sans hotspots)."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(8, 8))
    zone.boundary.plot(ax=ax, edgecolor="red", linewidth=1.8, zorder=3)
    minx, miny, maxx, maxy = zone.total_bounds
    dx, dy = (maxx - minx) * marge, (maxy - miny) * marge
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)
    if fond:
        _fond_osm(ax, zone.crs)
    ax.legend(handles=[Line2D([0], [0], color="red", lw=1.8, label="Zone d'étude")],
              loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Localisation de la zone d'étude")
    ax.set_xlabel("X (m, EPSG:32198)")
    ax.set_ylabel("Y (m)")
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return sortie


def fig_points_ebird(zone, couches: list[dict], sortie: str, titre: str, marge: float = 0.12,
                     fond=True) -> str:
    """Carte de points eBird sur fond OSM (distribution spatiale).

    `couches` : liste de dicts {x, y, couleur, taille?, alpha?, edge?, label}, tracés dans l'ordre.
    ⚠️ Données sensibles (espèce en péril + confidentialité eBird) : à écrire hors dépôt.
    """
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(8, 8))
    zone.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, zorder=2)
    handles = [Line2D([0], [0], color="black", lw=1.5, label="Zone d'étude")]
    for i, c in enumerate(couches):
        ax.scatter(np.asarray(c["x"]), np.asarray(c["y"]), s=c.get("taille", 18), color=c["couleur"],
                   edgecolors=c.get("edge", "none"), linewidths=0.3, alpha=c.get("alpha", 1.0),
                   zorder=3 + i)
        handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c["couleur"],
                              markeredgecolor=c.get("edge", "none"), markersize=7,
                              alpha=min(c.get("alpha", 1.0) + 0.4, 1.0), label=c["label"]))
    minx, miny, maxx, maxy = zone.total_bounds
    dx, dy = (maxx - minx) * marge, (maxy - miny) * marge
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)
    if fond:
        _fond_osm(ax, zone.crs)
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(titre)
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


def fig_shap(npz_path: str, sortie: str, max_display: int = 10) -> str:
    """Beeswarm SHAP (modèle habitat, classe présence) : contribution de chaque variable par checklist."""
    import shap
    d = np.load(npz_path, allow_pickle=True)
    noms = [ETIQUETTES.get(f, f) for f in d["features"].tolist()]
    shap.summary_plot(d["values"], d["data"], feature_names=noms, max_display=max_display,
                      show=False, plot_size=(8, 6), color_bar_label="Valeur de la variable")
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Valeur SHAP (→ probabilité de présence)")
    fig.suptitle("SHAP — contributions à la probabilité de présence (modèle habitat)", fontsize=11)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return sortie


def fig_shap_elevation(npz_path: str, sortie: str) -> str:
    """Dépendance SHAP de l'élévation : valeur SHAP vs élévation, présences/absences distinguées."""
    d = np.load(npz_path, allow_pickle=True)
    feats = d["features"].tolist()
    j = feats.index("elevation")
    elev, sv, pres = d["data"][:, j], d["values"][:, j], d["presence"].astype(bool)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.scatter(elev[~pres], sv[~pres], s=8, alpha=0.3, color="#4575b4", label="absence", zorder=2)
    ax.scatter(elev[pres], sv[pres], s=20, alpha=0.9, color="#d73027", label="présence", zorder=3)
    ax.set_xlabel(ETIQUETTES["elevation"])
    ax.set_ylabel("Valeur SHAP (→ probabilité de présence)")
    ax.set_title("Dépendance SHAP — élévation")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return sortie


def fig_diagnostic_elevation(diag: dict, sortie: str) -> str:
    """Diagnostic proxy spatial : importance de l'élévation sans/avec coordonnées + importance de x,y."""
    labels = ["élévation\n(sans x,y)", "élévation\n(avec x,y)", "x", "y"]
    vals = [diag["elevation_sans_coords"]["importance"], diag["elevation_avec_coords"]["importance"],
            diag["coords"]["x"]["importance"], diag["coords"]["y"]["importance"]]
    couleurs = ["#2c7fb8", "#7fcdbb", "#d95f0e", "#d95f0e"]
    fig, ax = plt.subplots(figsize=(7, 5))
    barres = ax.bar(labels, vals, color=couleurs)
    ax.set_ylabel("Importance par permutation (chute d'AUC-ROC)")
    ax.set_title("L'élévation est-elle un proxy spatial ?")
    for b, v in zip(barres, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    fontsize=8)
    txt = (f"R²(élévation ~ x,y) = {diag['r2_elevation_xy']:.2f}\n"
           f"AUC habitat : {diag['auc_habitat']:.3f} → {diag['auc_habitat_xy']:.3f} (avec x,y)")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
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
    if routes is not None and "ClsRte" in routes.columns:  # cartes = routes principales seulement
        routes = routes[routes["ClsRte"].isin(cfg.figures.routes_classes_principales)]
        log.info("  routes principales retenues : %d tronçons (%s)", len(routes),
                 ", ".join(cfg.figures.routes_classes_principales))
    hs_path = Path(out) / "tables" / "hotspots.gpkg"
    hotspots = gpd.read_file(hs_path) if hs_path.exists() else None
    produits: dict = {}

    with utils.log_step("Carte de localisation (fond OSM, sans hotspots)", log):
        produits["localisation"] = fig_localisation(zone, str(fig_dir / "localisation.png"))

    # Cartes de distribution eBird — DONNÉES SENSIBLES (espèce en péril + confidentialité eBird).
    # Écrites dans outputs/figures/prive/ (ignoré par Git) : jamais versionnées ni publiées.
    table = Path(cfg.chemins.processed) / "table_modele.parquet"
    if table.exists():
        import polars as pl
        prive = fig_dir / "prive"
        prive.mkdir(exist_ok=True)
        df = pl.read_parquet(table, columns=["x", "y", "presence", "minutes_apres_coucher"])
        pres = df.filter(pl.col("presence") == 1)
        absc = df.filter(pl.col("presence") == 0)
        col = lambda d, c: d[c].to_numpy()  # noqa: E731
        with utils.log_step("Cartes eBird observations/checklists [privé, hors dépôt]", log):
            produits["observations"] = fig_points_ebird(
                zone, [dict(x=col(pres, "x"), y=col(pres, "y"), couleur="#d73027", taille=26,
                            edge="white", label=f"Présences ({len(pres)})")],
                str(prive / "carte_observations.png"),
                f"Détections — {cfg.espece.nom_commun} (n={len(pres)})")
            produits["checklists"] = fig_points_ebird(
                zone,
                [dict(x=col(absc, "x"), y=col(absc, "y"), couleur="#4575b4", taille=4, alpha=0.25,
                      label=f"Sans détection ({len(absc)})"),
                 dict(x=col(pres, "x"), y=col(pres, "y"), couleur="#d73027", taille=16, edge="white",
                      label=f"Présences ({len(pres)})")],
                str(prive / "carte_checklists.png"),
                f"Checklists d'inventaire (n={len(df)})")

            # Checklists à une heure PERTINENTE pour un nocturne (fenêtre crépuscule → nuit).
            lo, hi = cfg.figures.fenetre_detection_apres_coucher_min
            fen = pl.col("minutes_apres_coucher").is_between(lo, hi)
            det, det_a, det_p = df.filter(fen), absc.filter(fen), pres.filter(fen)
            pct = 100 * len(det) / len(df)
            log.info("  fenêtre détection [%g, %g] min : %d checklists (%.1f%% de l'effort), "
                     "%d/%d présences", lo, hi, len(det), pct, len(det_p), len(pres))
            produits["checklists_detection"] = fig_points_ebird(
                zone,
                [dict(x=col(det_a, "x"), y=col(det_a, "y"), couleur="#4575b4", taille=10, alpha=0.5,
                      label=f"Sans détection ({len(det_a)})"),
                 dict(x=col(det_p, "x"), y=col(det_p, "y"), couleur="#d73027", taille=18,
                      edge="white", label=f"Présences ({len(det_p)})")],
                str(prive / "carte_checklists_detection.png"),
                f"Checklists en période de détection — n={len(det)} ({pct:.0f} % de l'effort)")

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

    shap_npz = Path(out) / "models" / "shap_habitat.npz"
    if shap_npz.exists():
        with utils.log_step("Figures SHAP (beeswarm + dépendance élévation)", log):
            produits["shap"] = fig_shap(str(shap_npz), str(fig_dir / "shap_habitat.png"))
            feats_npz = np.load(shap_npz, allow_pickle=True)["features"].tolist()
            if "elevation" in feats_npz:  # variante sans élévation → pas de dépendance
                produits["shap_elevation"] = fig_shap_elevation(
                    str(shap_npz), str(fig_dir / "shap_elevation.png"))
    else:
        log.warning("shap_habitat.npz absent — figures SHAP ignorées (relancer 03_model.py)")

    diag = rapport_model.get("diagnostic_elevation")
    if diag:
        with utils.log_step("Figure diagnostic élévation (proxy spatial)", log):
            produits["diagnostic_elevation"] = fig_diagnostic_elevation(
                diag, str(fig_dir / "diagnostic_elevation.png"))

    for nom, chemin in produits.items():
        log.info("  %s → %s", nom, chemin)
    utils.ecrire_rapport_json("05_figures", {"figures": produits}, log_dir=f"{out}/logs")


if __name__ == "__main__":
    main()
