"""02 — eBird : zero-fill auk → filtres d'effort → variables de détection → covariables.

Entrée : le zero-fill produit par `auk` (une ligne par checklist, listes de groupe déjà
fusionnées) — voir `config.ebird.zerofill_csv`. Le script applique les filtres d'effort
(Johnston et al. 2021 : Stationary/Traveling, listes complètes, durée ≤ 300 min,
distance ≤ 5 km, ≤ 10 observateurs), la saison (juin–juillet) et un plancher d'année
(covariables = instantané contemporain), restreint à l'emprise du stack, calcule les
variables de détection (minutes_apres_coucher, phase_lune, log_duree, jour_julien) et
extrait les 10 covariables du stack dans un buffer 30 m autour du point GPS.

Lecture lazy via polars.scan_csv ; collecte seulement après filtrage (jeu réduit).

Sortie : data/processed/table_modele.parquet
"""

from __future__ import annotations

import datetime
import logging
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

import utils
from config import Config, load_config_from_cli

log = logging.getLogger("02_ebird")

# Colonnes utiles du zero-fill (projection poussée par polars).
COLS_ZEROFILL = [
    "sampling_event_identifier", "scientific_name", "observation_date",
    "time_observations_started", "latitude", "longitude", "duration_minutes",
    "effort_distance_km", "number_observers", "protocol_name",
    "all_species_reported", "species_observed",
]
NA = "NA"  # valeur manquante du format EBD


# ── Emprise ──────────────────────────────────────────────────────────────────

def bbox_wgs84(stack_path: str | Path, marge_m: float = 2000.0) -> tuple[float, float, float, float]:
    """Emprise du stack (lon0, lat0, lon1, lat1) en WGS84, élargie de `marge_m`."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(stack_path) as ds:
        left, bottom, right, top = ds.bounds
        left, bottom, right, top = left - marge_m, bottom - marge_m, right + marge_m, top + marge_m
        return transform_bounds(ds.crs, "EPSG:4326", left, bottom, right, top)


# ── Filtres d'effort (Johnston et al. 2021) + saison + année + emprise ───────

def filtrer_checklists(lf: pl.LazyFrame, cfg: Config,
                       bbox: tuple[float, float, float, float]) -> pl.LazyFrame:
    """Applique filtres d'effort, saison, plancher d'année et emprise (LazyFrame → LazyFrame)."""
    eb = cfg.ebird
    lon0, lat0, lon1, lat1 = bbox
    lf = lf.with_columns(
        pl.col("observation_date").str.slice(0, 4).cast(pl.Int32).alias("annee"),
        pl.col("observation_date").str.slice(5, 2).cast(pl.Int32).alias("mois"),
        pl.col("duration_minutes").cast(pl.Float64, strict=False).alias("duree_min"),
        pl.when(pl.col("effort_distance_km") == NA).then(0.0)
          .otherwise(pl.col("effort_distance_km").cast(pl.Float64, strict=False)).alias("distance_km"),
        pl.col("number_observers").cast(pl.Float64, strict=False).alias("n_observateurs"),
        pl.col("latitude").cast(pl.Float64, strict=False).alias("lat"),
        pl.col("longitude").cast(pl.Float64, strict=False).alias("lon"),
        (pl.col("species_observed") == "TRUE").cast(pl.Int8).alias("presence"),
    )
    cond = (
        pl.col("mois").is_in(eb.mois_saison)
        & pl.col("protocol_name").is_in(eb.protocoles)
        & (pl.col("duree_min") <= eb.duree_max_min)
        & (pl.col("distance_km") <= eb.distance_max_km)
        & (pl.col("n_observateurs") <= eb.observateurs_max)
        & (pl.col("lon") >= lon0) & (pl.col("lon") <= lon1)
        & (pl.col("lat") >= lat0) & (pl.col("lat") <= lat1)
    )
    if eb.listes_completes:
        cond = cond & (pl.col("all_species_reported") == "TRUE")
    if eb.annee_min is not None:
        cond = cond & (pl.col("annee") >= eb.annee_min)
    return lf.filter(cond)


# ── Variables de détection (RF uniquement, non cartographiées) ───────────────

def jour_julien(date: datetime.date) -> int:
    """Jour de l'année (1–366)."""
    return date.timetuple().tm_yday


def log_duree(duree_min: float) -> float:
    """log(1 + durée en minutes) — comprime l'effet de l'effort d'écoute."""
    return float(np.log1p(duree_min))


def phase_lune(date: datetime.date) -> float:
    """Phase lunaire astral (0 = nouvelle lune … ~14 = pleine lune … 27,99)."""
    from astral import moon
    return float(moon.phase(date))


def _parse_heure(heure: str | None) -> datetime.time | None:
    """'HH:MM:SS' → datetime.time ; 'NA'/vide/None → None."""
    if not heure or heure == NA:
        return None
    try:
        h, m, *s = heure.split(":")
        return datetime.time(int(h), int(m), int(s[0]) if s else 0)
    except (ValueError, IndexError):
        return None


def minutes_apres_coucher(lat: float, lon: float, date: datetime.date,
                          heure: datetime.time | None, fuseau: str) -> float | None:
    """Minutes entre le coucher du soleil local (DST) et le début de la checklist.

    Repli cyclique sur [-720, 720) : positif = après le coucher (activité crépusculaire /
    nocturne), avec continuité de la nuit après minuit (00 h 30 → ~+230 min, non -1210).
    None si l'heure est absente ou si le soleil ne se couche pas (jamais au Québec).
    """
    if heure is None:
        return None
    from astral import Observer
    from astral.sun import sun
    tz = ZoneInfo(fuseau)
    try:
        coucher = sun(Observer(latitude=lat, longitude=lon), date=date, tzinfo=tz)["sunset"]
    except ValueError:
        return None
    debut = datetime.datetime.combine(date, heure, tzinfo=tz)
    delta = (debut - coucher).total_seconds() / 60.0
    return ((delta + 720.0) % 1440.0) - 720.0


def ajouter_variables_detection(df: pl.DataFrame, fuseau: str) -> pl.DataFrame:
    """Ajoute jour_julien, log_duree, phase_lune, minutes_apres_coucher (par ligne)."""
    from tqdm import tqdm

    dates = [datetime.date.fromisoformat(d) for d in df["observation_date"]]
    heures = [_parse_heure(h) for h in df["time_observations_started"]]
    lats, lons = df["lat"].to_list(), df["lon"].to_list()

    # phase lunaire : ne dépend que de la date → cache par date unique
    cache_lune = {d: phase_lune(d) for d in set(dates)}
    lune = [cache_lune[d] for d in dates]

    mac = [minutes_apres_coucher(la, lo, d, h, fuseau)
           for la, lo, d, h in tqdm(zip(lats, lons, dates, heures), total=len(dates),
                                     desc="minutes_apres_coucher", unit="chk")]
    return df.with_columns(
        pl.Series("jour_julien", [jour_julien(d) for d in dates], dtype=pl.Int32),
        pl.Series("log_duree", [log_duree(x) for x in df["duree_min"]], dtype=pl.Float64),
        pl.Series("phase_lune", lune, dtype=pl.Float64),
        pl.Series("minutes_apres_coucher", mac, dtype=pl.Float64),
    )


# ── Extraction des covariables (buffer 30 m sur le stack 5 m) ────────────────

def reprojeter_points(lon, lat, crs_cible: str) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat WGS84) → (x, y) dans `crs_cible` (vectorisé)."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs_cible, always_xy=True)
    x, y = tr.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def extraire_covariables_buffer(stack_path: str | Path, xs: np.ndarray, ys: np.ndarray,
                                buffer_m: float) -> tuple[np.ndarray, list[str]]:
    """Moyenne des covariables du stack dans un disque de rayon `buffer_m` autour de chaque point.

    Retourne (valeurs (n_points, n_bandes), noms des bandes). Pixels hors emprise = NaN
    (moyenne calculée sur les seuls pixels valides du disque ; tout-NaN → NaN).
    """
    import warnings

    import rasterio
    from rasterio.transform import rowcol
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
    from tqdm import tqdm

    xs, ys = np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")
    # cache GDAL large (1 GiB en octets — non ambigu) + points triés par bloc → réutilise
    # les blocs COG décompressés d'un point au suivant (évite de re-décompresser à chaque lecture)
    with rasterio.Env(GDAL_CACHEMAX=1073741824), rasterio.open(stack_path) as ds:
        n_bandes = ds.count
        noms = list(ds.descriptions)
        res_x, res_y = ds.res
        inv = ~ds.transform
        out = np.full((len(xs), n_bandes), np.nan, dtype="float32")
        # rayon en pixels (+1 de marge pour couvrir tout le disque après arrondi)
        rad_c = int(math.ceil(buffer_m / res_x)) + 1
        rad_r = int(math.ceil(buffer_m / res_y)) + 1
        rows, cols = (np.asarray(a) for a in rowcol(ds.transform, xs, ys))
        by, bx = ds.block_shapes[0]                            # (hauteur, largeur) de bloc COG
        ordre = np.lexsort((cols // bx, rows // by))          # tri par bloc (ligne puis colonne)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            for i in tqdm(ordre, desc="covariables (buffer 30 m)", unit="chk"):
                col_f, row_f = inv * (xs[i], ys[i])
                col0, row0 = int(math.floor(col_f)) - rad_c, int(math.floor(row_f)) - rad_r
                win = Window(col0, row0, 2 * rad_c + 1, 2 * rad_r + 1)
                data = ds.read(window=win, boundless=True, fill_value=np.nan)  # (b, h, w)
                tr = window_transform(win, ds.transform)
                rr, cc = np.mgrid[0:data.shape[1], 0:data.shape[2]]
                px = tr.c + (cc + 0.5) * tr.a + (rr + 0.5) * tr.b
                py = tr.f + (cc + 0.5) * tr.d + (rr + 0.5) * tr.e
                disque = (px - xs[i]) ** 2 + (py - ys[i]) ** 2 <= buffer_m ** 2
                if not disque.any():
                    continue
                vals = np.where(disque[None], data, np.nan).reshape(n_bandes, -1)
                out[i] = np.nanmean(vals, axis=1)
        return out, noms


# ── Orchestration ────────────────────────────────────────────────────────────

def charger_et_filtrer(cfg: Config, stack_path: str, rapport: dict) -> pl.DataFrame:
    """Charge le zero-fill, applique tous les filtres, collecte le jeu réduit."""
    with utils.log_step("Chargement + filtres (Johnston, saison, année, emprise)", log):
        bbox = bbox_wgs84(stack_path, marge_m=cfg.ebird.buffer_m)
        lf = pl.scan_csv(cfg.ebird.zerofill_csv, infer_schema_length=0).select(COLS_ZEROFILL)
        n_total = lf.select(pl.len()).collect().item()
        df = filtrer_checklists(lf, cfg, bbox).collect()
    log.info("  %d checklists sur %d après filtres (%d présences)",
             df.height, n_total, int(df["presence"].sum()))
    rapport.update(n_checklists_total=n_total, n_apres_filtres=df.height,
                   n_presences_filtres=int(df["presence"].sum()))
    return df


def extraire(cfg: Config, df: pl.DataFrame, stack_path: str,
             rapport: dict) -> tuple[pl.DataFrame, list[str]]:
    """Reprojette les points et extrait les 10 covariables du stack (buffer 30 m)."""
    with utils.log_step("Extraction des covariables (buffer 30 m)", log):
        xs, ys = reprojeter_points(df["lon"], df["lat"], cfg.zone_etude.crs)
        vals, noms = extraire_covariables_buffer(stack_path, xs, ys, cfg.ebird.buffer_m)
        df = df.with_columns(
            pl.Series("x", xs, dtype=pl.Float64), pl.Series("y", ys, dtype=pl.Float64),
            *[pl.Series(noms[j], vals[:, j], dtype=pl.Float32) for j in range(len(noms))],
        )
    rapport["bandes_covariables"] = noms
    return df, noms


def finaliser(df: pl.DataFrame, noms_cov: list[str], cfg: Config,
              stack_path: str, rapport: dict) -> pl.DataFrame:
    """Retire les cas incomplets (hors couverture / heure absente) et ordonne les colonnes."""
    n0 = df.height
    # nodata du stack = NaN flottant (pas null polars) → tester is_null OU is_nan
    cov_manquant = pl.any_horizontal([pl.col(c).is_null() | pl.col(c).is_nan() for c in noms_cov])
    hors_stack = int(df.select(cov_manquant.alias("m")).to_series().sum())
    sans_heure = int(df["minutes_apres_coucher"].is_null().sum())
    df = df.filter(~cov_manquant & pl.col("minutes_apres_coucher").is_not_null())

    detection = ["minutes_apres_coucher", "phase_lune", "log_duree", "jour_julien"]
    meta = ["sampling_event_identifier", "observation_date", "annee", "lat", "lon", "x", "y",
            "duree_min", "distance_km", "n_observateurs", "protocol_name", "presence"]
    df = df.select([*meta, *detection, *noms_cov])

    n_pres = int(df["presence"].sum())
    log.info("  table finale : %d checklists (%d présences, %d absences) ; "
             "retirés : %d hors stack, %d sans heure",
             df.height, n_pres, df.height - n_pres, int(hors_stack), sans_heure)
    rapport.update(
        n_hors_couverture_stack=int(hors_stack), n_sans_heure=sans_heure,
        n_final=df.height, n_presences=n_pres, n_absences=df.height - n_pres,
        ratio_presence=round(n_pres / df.height, 4) if df.height else None,
        annee_min=int(df["annee"].min()), annee_max=int(df["annee"].max()),
        taux_incomplets_pct=round((n0 - df.height) / n0 * 100, 2) if n0 else None,
        stack=stack_path,
    )
    return df


def main() -> None:
    cfg = load_config_from_cli()
    utils.setup_logging("02_ebird", log_dir=f"{cfg.chemins.outputs}/logs")
    rapport: dict = {}
    stack_path = f"{cfg.chemins.processed}/stack_5m.tif"
    if not Path(stack_path).exists():
        raise FileNotFoundError(f"Stack introuvable ({stack_path}) — exécuter 01_predictors.py (J3).")

    df = charger_et_filtrer(cfg, stack_path, rapport)
    with utils.log_step("Variables de détection", log):
        df = ajouter_variables_detection(df, cfg.ebird.fuseau)
    df, noms_cov = extraire(cfg, df, stack_path, rapport)
    df = finaliser(df, noms_cov, cfg, stack_path, rapport)

    sortie = Path(cfg.chemins.processed) / "table_modele.parquet"
    with utils.log_step(f"Écriture {sortie}", log):
        df.write_parquet(sortie)
    rapport["sortie"] = str(sortie)
    utils.ecrire_rapport_json("02_ebird", rapport, log_dir=f"{cfg.chemins.outputs}/logs")


if __name__ == "__main__":
    main()
