"""09 — Figures : caractéristiques d'habitat des hotspots (avec vs sans élévation).

Chaque méthode (modèle combiné avec / sans élévation) désigne ses propres cellules
« hotspot » (1 km, haute probabilité × faible effort eBird, J6). Ce script extrait les
10 variables d'habitat du `stack_5m.tif` **à l'intérieur de chaque cellule hotspot** et
compare les deux jeux de hotspots entre eux et au fond du territoire d'étude.

Figures (outputs/comparaison/figures/) :
- hotspots_profil_habitat.png   — profil d'écart au paysage (z-scores), avec vs sans
- hotspots_distributions.png    — distributions par variable (fond · avec · sans)
- hotspots_carte_elevation.png  — position spatiale des hotspots sur l'élévation

Usage : `uv run python code/09_figures_hotspots.py`
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "code"))
log = logging.getLogger("09_figures_hotspots")

# ── Palette colorblind-safe (Okabe-Ito) — cohérente avec 08_figures_rapport ──
C_AVEC, C_SANS, C_FOND = "#0072B2", "#D55E00", "#B7C4CE"
C_PRES = "#009E73"                              # présences réelles (Okabe-Ito bleu-vert)
INK, MUTED, GRID = "#1a1a1a", "#5f5f5f", "#e8e8e8"

STACK = RACINE / "data/processed/stack_5m.tif"
TABLE = RACINE / "data/processed/table_modele.parquet"
BANDS = ["MHC_hauteur_canopee", "TWI", "densite_lisiere", "prop_feuillu_melange",
         "densite_routes", "LST_estivale", "classe_age", "densite_peuplement",
         "distance_milieu_humide", "elevation"]
# Étiquette courte + unité (pour les axes en valeurs réelles)
ETIQ = {
    "MHC_hauteur_canopee": ("Hauteur canopée", "m"),
    "TWI": ("TWI", "indice"),
    "densite_lisiere": ("Densité de lisière", "prop. focal 1 km"),
    "prop_feuillu_melange": ("Prop. feuillu/mélangé", "0–1"),
    "densite_routes": ("Densité de routes", "km/km²"),
    "LST_estivale": ("LST estivale", "°C"),
    "classe_age": ("Classe d'âge", "0–4"),
    "densite_peuplement": ("Densité peuplement", "% couvert"),
    "distance_milieu_humide": ("Distance milieu humide", "m"),
    "elevation": ("Élévation", "m"),
}
HOTSPOTS = {
    "avec": RACINE / "outputs/avec_elevation/tables/hotspots.gpkg",
    "sans": RACINE / "outputs/sans_elevation/tables/hotspots.gpkg",
}
RNG = np.random.default_rng(42)


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
                         "axes.titlesize": 11, "figure.dpi": 150})


def _sans_cadre(ax, garder=("left", "bottom")) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in garder)


# ── Extraction des covariables dans les cellules hotspot ─────────────────────

def _fond(n_side: int = 700) -> np.ndarray:
    """Échantillon paysager : lecture décimée (nearest) du stack → (n_bandes, n_px valides)."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(STACK) as src:
        bg = src.read(out_shape=(src.count, n_side, n_side),
                      resampling=Resampling.nearest).astype("float32")
    bg = bg.reshape(len(BANDS), -1)
    return bg[:, np.isfinite(bg).all(axis=0)]


def _extraire_cellules(gpkg: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Pour chaque hotspot : moyennes par bande (n_cell, n_bande) + pool de pixels valides."""
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds
    gdf = gpd.read_file(gpkg).sort_values("rang")
    moyennes, pools, meta = [], [], []
    with rasterio.open(STACK) as src:
        for _, row in gdf.iterrows():
            win = from_bounds(*row.geometry.bounds, transform=src.transform)
            win = win.round_lengths().round_offsets()
            arr = src.read(window=win).astype("float32").reshape(len(BANDS), -1)
            val = np.isfinite(arr).all(axis=0)
            px = arr[:, val]
            moyennes.append(px.mean(axis=1))
            pools.append(px)
            meta.append({"rang": int(row["rang"]), "proba": float(row["proba"]),
                         "x": float(row["x_centre"]), "y": float(row["y_centre"])})
    return np.asarray(moyennes), np.hstack(pools), meta


def _extraire_points(xy: list[tuple[float, float]], taille_m: float = 1000.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Habitat dans une tuile centrée (`taille_m`) autour de chaque observation réelle.

    Même gabarit que les cellules hotspot → comparaison à gabarit égal. Retourne les
    moyennes par point (n_pts, n_bande) et le pool de pixels valides.
    """
    import rasterio
    from rasterio.windows import from_bounds
    half = taille_m / 2
    moyennes, pools = [], []
    with rasterio.open(STACK) as src:
        for x, y in xy:
            win = from_bounds(x - half, y - half, x + half, y + half, transform=src.transform)
            win = win.round_lengths().round_offsets()
            arr = src.read(window=win, boundless=True, fill_value=np.nan)
            arr = arr.astype("float32").reshape(len(BANDS), -1)
            val = np.isfinite(arr).all(axis=0)
            if not val.any():
                continue
            px = arr[:, val]
            moyennes.append(px.mean(axis=1))
            pools.append(px)
    return np.asarray(moyennes), np.hstack(pools)


def _sous_ech(px: np.ndarray, n: int = 6000) -> np.ndarray:
    """Sous-échantillonne les colonnes (pixels) d'un pool (n_bandes, N) à n colonnes."""
    if px.shape[1] <= n:
        return px
    return px[:, RNG.choice(px.shape[1], n, replace=False)]


# ── Fig 1 — Profil d'habitat standardisé (écart au paysage, z-scores) ─────────

def fig_profil(cells: dict, bg_mu: np.ndarray, bg_sd: np.ndarray, sortie: Path) -> dict:
    """Barres divergentes avec/sans + cible « présences réelles » (losange). Retourne les distances au profil réel."""
    from matplotlib.lines import Line2D
    z = {m: (cells[m] - bg_mu) / bg_sd for m in cells}                 # (n_unit, n_bande)
    mu = {m: z[m].mean(axis=0) for m in z}
    se = {m: z[m].std(axis=0, ddof=1) / np.sqrt(z[m].shape[0]) for m in z}
    ordre = np.argsort(np.abs(mu["pres"]))                            # croissant → haut = signal réel fort
    labels = [f"{ETIQ[BANDS[i]][0]}" for i in ordre]
    dist = {m: float(np.mean(np.abs(mu[m] - mu["pres"]))) for m in ("avec", "sans")}

    y, h = np.arange(len(ordre)), 0.36
    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    xmax = 0.0
    for m, coul, lab, dy in [("avec", C_AVEC, "Hotspots avec élévation", h / 2),
                             ("sans", C_SANS, "Hotspots sans élévation", -h / 2)]:
        vals, errs = mu[m][ordre], se[m][ordre]
        ax.barh(y + dy, vals, h, xerr=errs, color=coul, label=lab,
                error_kw={"elinewidth": 1.0, "ecolor": MUTED, "capsize": 2.5})
        for yi, v, e in zip(y + dy, vals, errs):                        # valeur hors moustache
            dx = e + 0.12
            ax.text(v + dx if v >= 0 else v - dx, yi, f"{v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.4, color=INK)
            xmax = max(xmax, abs(v) + e)
    # Présences réelles = cible (losange + barre d'erreur), même standardisation
    vp, ep = mu["pres"][ordre], se["pres"][ordre]
    ax.errorbar(vp, y, xerr=ep, fmt="D", color=C_PRES, markersize=6.5, markeredgecolor="white",
                markeredgewidth=0.6, ecolor=C_PRES, elinewidth=1.1, capsize=2.5, zorder=5)
    xmax = max(xmax, float(np.max(np.abs(vp) + ep)))
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-xmax * 1.32, xmax * 1.32)
    ax.set_xlabel("Écart au paysage d'étude  (nombre d'écarts-types ; 0 = moyenne du territoire)")
    ax.set_title("Habitat des hotspots vs présences réelles — avec vs sans élévation", fontsize=11.5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    _sans_cadre(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_AVEC), plt.Rectangle((0, 0), 1, 1, color=C_SANS),
               Line2D([0], [0], marker="D", color="none", markerfacecolor=C_PRES,
                      markeredgecolor="white", markersize=8)]
    ax.legend(handles, ["Hotspots avec élévation", "Hotspots sans élévation",
                        f"Présences réelles (n={cells['pres'].shape[0]})"],
              loc="lower right", frameon=False, fontsize=9)
    plus_proche = "avec" if dist["avec"] < dist["sans"] else "sans"
    fig.text(0.5, -0.02, f"Losange = profil des présences réelles (cible) · écart moyen au profil réel : "
             f"avec {dist['avec']:.2f} σ · sans {dist['sans']:.2f} σ  →  « {plus_proche} » plus proche · "
             "tuiles 1 km, variables triées par signal réel", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(sortie, bbox_inches="tight")
    plt.close(fig)
    log.info("  écrit %s (écart avec=%.2f σ, sans=%.2f σ)", sortie.name, dist["avec"], dist["sans"])
    return dist


# ── Fig 2 — Distributions par variable (paysage · présences · avec · sans) ────

def fig_distributions(cells: dict, pools: dict, bg: np.ndarray, sortie: Path) -> None:
    """Petits multiples : boxplot des pixels (paysage/présences/avec/sans) + moyennes par hotspot."""
    grp = [("Fond", C_FOND, bg), ("pres", C_PRES, pools["pres"]),
           ("avec", C_AVEC, pools["avec"]), ("sans", C_SANS, pools["sans"])]
    ech = {nom: _sous_ech(px) for nom, _, px in grp}
    pos = {"Fond": 1, "pres": 2, "avec": 3, "sans": 4}
    fig, axes = plt.subplots(2, 5, figsize=(16.8, 7.4))
    for k, (ax, b) in enumerate(zip(axes.ravel(), BANDS)):
        donnees = [ech[nom][k] for nom, _, _ in grp]
        bp = ax.boxplot(donnees, positions=list(pos.values()), widths=0.62, patch_artist=True,
                        showfliers=False, medianprops={"color": INK, "lw": 1.2},
                        whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for patch, (_, coul, _) in zip(bp["boxes"], grp):
            patch.set_facecolor(coul)
            patch.set_edgecolor(MUTED if coul == C_FOND else coul)
            patch.set_alpha(0.55)
        for m, coul in [("avec", C_AVEC), ("sans", C_SANS)]:              # moyennes par hotspot
            pts = cells[m][:, k]
            ax.scatter(np.full(pts.shape, pos[m]) + RNG.uniform(-0.13, 0.13, pts.shape), pts,
                       s=13, color=coul, edgecolor="white", linewidth=0.4, zorder=3)
        nom, unite = ETIQ[b]
        ax.set_title(nom, fontsize=9.8)
        ax.set_ylabel(unite, fontsize=8, color=MUTED)
        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels(["Paysage", "Prés.", "Avec", "Sans"], fontsize=7.8)
        ax.tick_params(axis="y", labelsize=7.6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        _sans_cadre(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.7) for _, c, _ in grp]
    fig.legend(handles, ["Paysage d'étude (fond)", "Présences réelles (tuiles 1 km)",
                         "Hotspots — avec élévation", "Hotspots — sans élévation"],
               loc="lower center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Conditions d'habitat : présences réelles vs hotspots des deux méthodes (tuiles 1 km)",
                 fontsize=13, y=1.0)
    fig.text(0.5, 0.955, "Boîtes = distribution des pixels 5 m · points = moyenne de chacun "
             "des 8 hotspots", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(sortie, bbox_inches="tight")
    plt.close(fig)
    log.info("  écrit %s", sortie.name)


# ── Fig 3 — Position spatiale des hotspots sur l'élévation ────────────────────

def fig_carte(meta: dict, sortie: Path) -> None:
    """Fond d'élévation (décimé) + cellules hotspot des deux méthodes → clustering spatial."""
    import geopandas as gpd
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.plot import plotting_extent
    i_elev = BANDS.index("elevation")
    with rasterio.open(STACK) as src:
        elev = src.read(i_elev + 1, out_shape=(900, 900),
                        resampling=Resampling.average).astype("float32")
        ext = plotting_extent(src)
    elev[~np.isfinite(elev)] = np.nan

    fig, ax = plt.subplots(figsize=(8.8, 8.4))
    ax.set_facecolor("#f7f1e3")                                          # nodata (nan) → crème, ≠ basse élévation
    im = ax.imshow(elev, extent=ext, origin="upper", cmap="Greys",
                   vmin=np.nanpercentile(elev, 2), vmax=np.nanpercentile(elev, 98))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Élévation (m)", fontsize=9)
    zone = RACINE / "data/zone_etude.gpkg"
    if zone.exists():
        gpd.read_file(zone).boundary.plot(ax=ax, color=MUTED, linewidth=0.8)
    for m, coul, lab in [("avec", C_AVEC, "Avec élévation"), ("sans", C_SANS, "Sans élévation")]:
        gdf = gpd.read_file(HOTSPOTS[m])
        gdf.boundary.plot(ax=ax, color=coul, linewidth=1.8)
        for _, r in gdf.iterrows():
            ax.annotate(str(int(r["rang"])), (r["x_centre"], r["y_centre"]), color=coul,
                        fontsize=8.5, weight="bold", ha="center", va="center")
        ax.plot([], [], color=coul, linewidth=2.2, label=lab)             # proxy légende
    ax.set_title("Où tombent les hotspots — élévation en fond", fontsize=11.5)
    ax.set_xlabel("X (m, EPSG:32198)", fontsize=8.5)
    ax.set_ylabel("Y (m, EPSG:32198)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)
    e_a = np.array([meta["avec"][i]["elev"] for i in range(len(meta["avec"]))])
    e_s = np.array([meta["sans"][i]["elev"] for i in range(len(meta["sans"]))])
    fig.text(0.5, -0.01, f"Élévation moyenne des hotspots — avec : {e_a.mean():.0f} m "
             f"(σ {e_a.std():.0f}) · sans : {e_s.mean():.0f} m (σ {e_s.std():.0f})",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(sortie, bbox_inches="tight")
    plt.close(fig)
    log.info("  écrit %s", sortie.name)


def main() -> None:
    import utils
    utils.setup_logging("09_figures_hotspots", log_dir="outputs/comparaison/logs")
    _style()
    comp = RACINE / "outputs/comparaison/figures"
    comp.mkdir(parents=True, exist_ok=True)

    with utils.log_step("Échantillon paysager (fond)", log):
        bg = _fond()
        bg_mu, bg_sd = bg.mean(axis=1), bg.std(axis=1)
        log.info("  %d pixels de fond", bg.shape[1])

    cells, pools, meta = {}, {}, {}
    with utils.log_step("Extraction des covariables dans les hotspots", log):
        i_elev = BANDS.index("elevation")
        for m, gpkg in HOTSPOTS.items():
            cells[m], pools[m], meta[m] = _extraire_cellules(gpkg)
            for i in range(len(meta[m])):
                meta[m][i]["elev"] = float(cells[m][i, i_elev])
            log.info("  %s : %d hotspots · %d px poolés", m, cells[m].shape[0], pools[m].shape[1])

    with utils.log_step("Extraction autour des présences réelles (tuiles 1 km)", log):
        import polars as pl
        dfp = pl.read_parquet(TABLE).filter(pl.col("presence") == 1)
        xy = list(zip(dfp["x"].to_list(), dfp["y"].to_list()))
        cells["pres"], pools["pres"] = _extraire_points(xy, taille_m=1000.0)
        log.info("  %d présences · %d px poolés", cells["pres"].shape[0], pools["pres"].shape[1])

    with utils.log_step("Fig 1 — profil d'habitat standardisé (vs présences)", log):
        dist = fig_profil(cells, bg_mu, bg_sd, comp / "hotspots_profil_habitat.png")
    with utils.log_step("Fig 2 — distributions par variable", log):
        fig_distributions(cells, pools, bg, comp / "hotspots_distributions.png")
    with utils.log_step("Fig 3 — carte des hotspots sur l'élévation", log):
        fig_carte(meta, comp / "hotspots_carte_elevation.png")

    rapport = {
        "n_hotspots": {m: int(cells[m].shape[0]) for m in ("avec", "sans")},
        "n_presences": int(cells["pres"].shape[0]),
        "elevation_moy_m": {m: round(float(np.mean([d["elev"] for d in meta[m]])), 1)
                            for m in meta},
        "elevation_std_m": {m: round(float(np.std([d["elev"] for d in meta[m]])), 1)
                            for m in meta},
        "ecart_moyen_au_profil_presences_sigma": {m: round(dist[m], 3) for m in dist},
        "modele_plus_proche_des_presences": min(dist, key=dist.get),
        "moyennes_habitat": {m: {BANDS[k]: round(float(cells[m][:, k].mean()), 3)
                                 for k in range(len(BANDS))} for m in cells},
        "figures": ["hotspots_profil_habitat.png", "hotspots_distributions.png",
                    "hotspots_carte_elevation.png"],
    }
    utils.ecrire_rapport_json("09_figures_hotspots", rapport, log_dir="outputs/comparaison/logs")


if __name__ == "__main__":
    main()
