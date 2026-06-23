# GMQ405 — Protocole méthodologique Python
## SDM Engoulevent bois-pourri — implémentation Python

**Version :** 1.0 — 2026-06-11
**Référence :** remplace l'implémentation R du plan `03_plan_projet_session.md`. Question de recherche, variables, hypothèses et planning hebdomadaire inchangés.

---

## 1. Stack technique

| Rôle | R (plan initial) | Python (équivalent retenu) | Justification |
|---|---|---|---|
| Manipulation tabulaire | `dplyr` / `data.table` | `pandas` / `polars` | `polars` pour l'EBD (~GB, lazy scan) ; `pandas` ailleurs |
| Vecteur | `sf` | `geopandas` + `shapely` | Standard de fait |
| Raster | `terra` | `rasterio` + `rioxarray` + `xarray` | `rioxarray` pour aligner stack + extraction |
| Reprojection / extent | `terra::project` | `rasterio.warp` / `rioxarray.reproject_match` | |
| Métriques paysagères | `landscapemetrics` | `pylandstats` | densité de lisière, Shannon |
| Filtrage eBird | `auk` | parsing manuel (`polars.scan_csv`) | pas d'équivalent direct → fonction maison |
| Zero-filling eBird | `auk_zerofill` | fonction maison | rejoindre EBD et sampling sur `SAMPLING EVENT IDENTIFIER` |
| MaxEnt | `maxnet` | `elapid` | implémentation Python pure de Maxnet (Phillips 2017) |
| Random Forest | `ranger` | `scikit-learn` (`RandomForestClassifier`) + `imbalanced-learn` | down-sampling via `BalancedRandomForestClassifier` |
| Validation spatiale | `blockCV` | `verde.BlockKFold` ou splitter maison | blocs spatiaux indépendants |
| Métriques | `pROC`, `ecospat` | `sklearn.metrics` + fonction CBI maison | AUC, TSS faciles ; CBI à coder (Hirzel 2006) |
| MESS | `dismo::mess` | fonction maison (~30 lignes) | détection extrapolation |
| Soleil / lune | `suncalc` | `astral` | heures crépuscule, phase lunaire |
| Plots / cartes | `ggplot2` / `tmap` | `matplotlib` + `contextily` | cartes raster simples |
| Rapport | `Quarto` | `Quarto` (kernel Python) ou `Jupyter` + nbconvert | `Quarto` recommandé (mêmes outputs) |
| Repro | `renv` | `uv` + `pyproject.toml` (ou `conda` env) | `uv` rapide et déterministe |

**Environnement** : Python 3.11+, gestion par `uv` (ou `mamba` si LiDAR exige GDAL non-pip).

---

## 2. Arborescence du projet

```
projet_sdm_engoulevent/
├── pyproject.toml              # dépendances + métadonnées (uv)
├── uv.lock                     # lockfile reproductible
├── README.md
├── .gitignore                  # exclut data/raw, outputs/
│
├── data/
│   ├── raw/                    # téléchargements bruts (déjà existant)
│   │   ├── ebird/
│   │   ├── MNT/  MHC/  Pentes/
│   │   ├── chelsa/
│   │   ├── ecoforestiere/
│   │   └── routes/
│   ├── interim/                # nettoyages intermédiaires
│   └── processed/              # produits finaux (stack raster, table modèle)
│       ├── stack_250m.tif      # multi-bandes alignées
│       ├── presences.gpkg
│       └── table_modele.parquet
│
├── src/
│   └── sdm_engoulevent/        # package importable
│       ├── __init__.py
│       ├── config.py           # constantes : EPSG, résolution, dates, chemins
│       ├── io_utils.py         # lecture/écriture raster + vecteur
│       ├── ebird.py            # filtrage, zero-fill, détection
│       ├── predictors.py       # construction des couches environnementales
│       ├── stack.py            # alignement + extraction valeurs aux points
│       ├── modeling.py         # MaxEnt + RF, tuning, importance
│       ├── validation.py       # blockCV, AUC/TSS/CBI, MESS
│       ├── prediction.py       # carte prédictive, bootstrap
│       └── viz.py              # figures et cartes
│
├── scripts/                    # scripts d'orchestration (numérotés)
│   ├── 00_setup.py             # vérifie env, crée dossiers
│   ├── 01_select_mrc.py
│   ├── 02_download_donnees.py  # déjà fait → telechargement_donnees.py
│   ├── 03_filter_ebird.py
│   ├── 04_prepare_predictors.py
│   ├── 05_build_stack.py
│   ├── 06_extract_covariates.py
│   ├── 07_vif_correlation.py
│   ├── 08_fit_maxent.py
│   ├── 09_fit_rf.py
│   ├── 10_block_cv.py
│   ├── 11_predict_maps.py
│   └── 12_figures.py
│
├── notebooks/                  # exploration interactive (pas de logique de prod)
│   └── exploration.ipynb
│
├── outputs/
│   ├── figures/
│   ├── maps/                   # GeoTIFF + PNG
│   ├── tables/
│   └── models/                 # joblib pickle (RF), parquet (params MaxEnt)
│
└── rapport/
    └── rapport.qmd             # Quarto, kernel Python
```

**Règle d'or** : la logique métier vit dans `src/sdm_engoulevent/` (testable, importable). Les scripts `scripts/` sont des entrées CLI courtes (~50 lignes) qui appellent le package et persistent les sorties.

---

## 3. Étapes opérationnelles et fonctions associées

Notation : `module.fonction(args) -> retour` — description.

### Étape 1 — Configuration globale (`config.py`)

Constantes partagées — pas de fonction, juste des valeurs :

- `CRS_TRAVAIL = "EPSG:32198"` (NAD83 / Quebec Lambert)
- `RESOLUTION_M = 250`
- `PERIODE_NIDIF = ("06-01", "07-31")` (juin–juillet)
- `MRC_NOM = "<à définir S1>"`
- `BUFFER_BACKGROUND_KM = 50`
- `SEED = 42`
- Chemins `DATA_RAW`, `DATA_PROCESSED`, `OUTPUTS` (via `pathlib`)

### Étape 2 — I/O utilitaires (`io_utils.py`)

- `lire_raster(path, masked=True) -> xarray.DataArray` — wrapper `rioxarray.open_rasterio` avec masque NoData.
- `ecrire_raster(da, path, compress="DEFLATE") -> None` — écrit avec tags + compression.
- `lire_vecteur(path, layer=None) -> gpd.GeoDataFrame` — autodétecte gpkg/shp.
- `reprojeter_vers_grille(src, grille_ref) -> xarray.DataArray` — `reproject_match` avec rééchantillonnage (bilinéaire pour continu, nearest pour catégoriel).
- `clip_vers_zone(da_ou_gdf, zone_gdf) -> mêmetype` — clip vectoriel ou raster.

### Étape 3 — Sélection MRC (`scripts/01_select_mrc.py`)

- `compter_presences_par_mrc(ebird_df, mrc_gdf) -> pd.DataFrame` — spatial join + comptage des observations Engoulevent par MRC.
- `evaluer_couverture_lidar(mrc_gdf, lidar_index_gdf) -> pd.DataFrame` — proportion de couverture LiDAR par MRC.
- `proposer_mrc_candidates(presences_par_mrc, couverture_lidar, n=3) -> pd.DataFrame` — classe par score composite (≥ seuil présences ET couverture LiDAR ≥ 90%).

**Livrable** : `outputs/tables/mrc_candidates.csv` + carte exploratoire.

### Étape 4 — Filtrage eBird (`ebird.py`)

L'EBD étant volumineux (~50 GB brut, ~GB après filtre QC), utiliser `polars` en mode lazy.

- `charger_ebd_polars(path) -> pl.LazyFrame` — `scan_csv` avec dtype mapping ; sépare EBD (observations) du sampling (checklists).
- `filtrer_espece(lf, espece="Eastern Whip-poor-will") -> pl.LazyFrame` — filtre COMMON NAME.
- `filtrer_periode(lf, mois=(6,7), annees=(2010,2025)) -> pl.LazyFrame` — fenêtre temporelle nidification.
- `filtrer_protocoles(lf, protocoles=("Stationary","Traveling")) -> pl.LazyFrame`.
- `filtrer_listes_completes(lf) -> pl.LazyFrame` — `ALL SPECIES REPORTED == 1`.
- `appliquer_filtres_johnston(lf, dur_max=300, dist_max_km=5, obs_max=10) -> pl.LazyFrame` — filtres Johnston 2021.
- `zero_fill(ebd_obs, ebd_sampling, espece) -> pd.DataFrame` — left-join checklists × observations sur `SAMPLING EVENT IDENTIFIER`, génère colonne `presence` binaire (0/1).
- `recentrer_heure_coucher_soleil(df, lat_col, lon_col, date_col, heure_col) -> pd.Series` — minutes après coucher du soleil via `astral`.
- `phase_lunaire(df, date_col) -> pd.Series` — fraction éclairée via `astral.moon`.

**Livrable** : `data/processed/checklists_zerofilled.parquet` avec colonnes `lat, lon, date, presence, DURATION_MINUTES, ...`.

### Étape 5 — Prédicteurs environnementaux (`predictors.py`)

Une fonction par variable, chacune retourne un `xarray.DataArray` projeté en `CRS_TRAVAIL`, résolution 250 m, aligné sur la grille de la MRC.

- `construire_grille_reference(mrc_gdf, resolution_m=250) -> xarray.DataArray` — raster vide qui sert d'aligneur (template).
- `var_mhc(tuiles_dir, grille_ref) -> xarray.DataArray` — mosaïque des feuillets MHC, agrégation `mean` à 250 m, reproject_match.
- `var_twi(tuiles_dir, grille_ref) -> xarray.DataArray` — idem TWI.
- `var_lisiere_foret_ouvert(ecoforestiere_gdf, grille_ref, fenetre_m=1000) -> xarray.DataArray` — rasterise classes forêt/ouvert, calcule densité de lisière avec `pylandstats` (focal_metric `edge_density`) en fenêtre 1 km.
- `var_prop_feuillu_melange(ecoforestiere_gdf, grille_ref, fenetre_m=500) -> xarray.DataArray` — proportion focale.
- `var_densite_routes(routes_gdf, grille_ref, fenetre_m=1000) -> xarray.DataArray` — rasterise lignes en km, focal sum / aire.
- `var_bio10(chelsa_path, grille_ref) -> xarray.DataArray` — clip + reproject.

Chaque fonction écrit son raster intermédiaire dans `data/interim/`.

### Étape 6 — Stack et extraction (`stack.py`)

- `assembler_stack(dict_da) -> xarray.Dataset` — empile les variables (mêmes coords) en un Dataset, écrit `stack_250m.nc` ou `.tif` multi-bandes.
- `extraire_aux_points(stack, points_gdf, methode="nearest") -> pd.DataFrame` — extraction vectorisée via `xarray.sel` ou `rasterio.sample`.
- `generer_background(zone_gdf, n=10000, biais_raster=None, seed=SEED) -> gpd.GeoDataFrame` — tirage pondéré par densité de checklists pour corriger biais d'effort.

### Étape 7 — Sélection de variables (`scripts/07_vif_correlation.py`)

- `matrice_correlation(df_covars) -> pd.DataFrame` — Spearman.
- `calculer_vif(df_covars) -> pd.Series` — boucle `statsmodels.stats.outliers_influence.variance_inflation_factor`.
- `eliminer_redondance(df_covars, seuil_corr=0.7, seuil_vif=5) -> list[str]` — élimination itérative ; conserve la variable la plus interprétable.

### Étape 8 — MaxEnt (`modeling.py`)

- `preparer_donnees_maxent(presences_df, background_df, covars) -> tuple[X, y]` — concat + flag presence/background.
- `tuner_maxent(X, y, grille_params, k_folds=5) -> dict` — grid search sur `feature_types` (`linear`, `quadratic`, `hinge`) × `regularization` (0.5, 1, 2, 4), sélection AICc.
- `ajuster_maxent(X, y, params) -> elapid.MaxentModel` — fit final.
- `importance_permutation(model, X, y, n_repeats=10) -> pd.DataFrame` — `sklearn.inspection.permutation_importance`.
- `courbes_reponse(model, X, var, n=100) -> pd.DataFrame` — fait varier `var` sur sa plage, fige les autres à la médiane.

### Étape 9 — Random Forest (`modeling.py`)

- `preparer_donnees_rf(table_zerofill, covars_occupation, covars_detection) -> tuple[X, y]` — inclut variables de détection.
- `tuner_rf(X, y, cv, grille_params) -> dict` — RandomizedSearchCV sur `n_estimators`, `max_features`, `min_samples_leaf`, `max_depth` ; scoring AUC.
- `ajuster_rf_balanced(X, y, params) -> BalancedRandomForestClassifier` — down-sampling de la classe majoritaire (`imblearn`).
- `importance_rf(model, X, y) -> pd.DataFrame` — permutation (préférée à Gini, biaisée).

### Étape 10 — Validation spatiale (`validation.py`)

- `bloc_spatial_kfold(points_gdf, taille_bloc_m, k=5, seed=SEED) -> Iterator[tuple[idx_train, idx_test]]` — découpe l'enveloppe en grille de blocs, assigne aléatoirement chaque bloc à un fold, retourne indices.
- `estimer_taille_bloc(residus_gdf) -> float` — semivariogramme avec `skgstat` ; range = taille minimale du bloc.
- `cv_evaluer(model_factory, X, y, points_gdf, taille_bloc) -> pd.DataFrame` — boucle k-folds, retourne AUC/TSS/CBI par fold.
- `auc_roc(y_true, y_pred) -> float` — `sklearn.metrics.roc_auc_score`.
- `tss(y_true, y_pred, seuil=0.5) -> float` — sensibilité + spécificité − 1.
- `cbi(y_true, y_pred, n_bins=10) -> float` — Boyce index continu (Hirzel 2006), corrélation de Spearman entre fréquence prédite/attendue par classe.

### Étape 11 — Cartographie prédictive (`prediction.py`)

- `predire_sur_stack(model, stack, batch_size=1_000_000) -> xarray.DataArray` — reshape stack en table, prédit par chunks (économie mémoire), reshape inverse.
- `bootstrap_incertitude(model_factory, X, y, stack, n_boot=100) -> xarray.Dataset` — refit sur échantillons bootstrap, retourne moyenne + écart-type pixel par pixel.
- `mess(stack_train, stack_predict) -> xarray.DataArray` — Multivariate Environmental Similarity Surface (Elith 2010) : pour chaque pixel, similarité min sur les variables vs distribution d'entraînement.
- `identifier_hotspots(carte_proba, carte_effort, top_n=5, seuil_proba=0.7) -> gpd.GeoDataFrame` — zones haute probabilité × faible effort eBird, retourne polygones + coords centroïdes.

### Étape 12 — Visualisations (`viz.py`)

- `plot_courbes_reponse(courbes_df, output_path) -> None` — facette par variable.
- `plot_importance(importance_df, output_path) -> None` — barplot trié.
- `plot_validation(cv_df, output_path) -> None` — boxplots AUC/TSS/CBI par modèle.
- `carte_probabilite(da, output_path, basemap=True) -> None` — `matplotlib` + `contextily`, colormap viridis, légende, échelle.
- `carte_desaccord(da_maxent, da_rf, output_path) -> None` — différence absolue.
- `carte_mess(da_mess, output_path) -> None` — bicolore (intérieur vs extrapolation).

---

## 4. Flux d'exécution (Makefile ou scripts orchestrés)

Convention : chaque script lit ses entrées dans `data/processed/` ou `outputs/`, écrit ses sorties au même endroit, et est idempotent.

```
00_setup.py            # vérifie versions + crée dossiers
01_select_mrc.py       # → outputs/tables/mrc_candidates.csv
02_download_donnees.py # → data/raw/  (déjà existant)
03_filter_ebird.py     # → data/processed/checklists_zerofilled.parquet
04_prepare_predictors.py # → data/interim/var_*.tif
05_build_stack.py      # → data/processed/stack_250m.nc
06_extract_covariates.py # → data/processed/table_modele.parquet
07_vif_correlation.py  # → outputs/tables/vif.csv (filtre variables)
08_fit_maxent.py       # → outputs/models/maxent.pkl + courbes + importance
09_fit_rf.py           # → outputs/models/rf.pkl + courbes + importance
10_block_cv.py         # → outputs/tables/cv_metriques.csv
11_predict_maps.py     # → outputs/maps/proba_{maxent,rf}.tif + mess + bootstrap
12_figures.py          # → outputs/figures/*.png
```

Optionnel : un `Makefile` rend la chaîne déclarative et permet `make all` ; sinon un script `run_all.py` séquentiel.

---

## 5. Tests et qualité

- **Tests unitaires** (`pytest`, dossier `tests/`) sur les fonctions à logique non triviale : `zero_fill`, `cbi`, `bloc_spatial_kfold`, `mess`, `recentrer_heure_coucher_soleil`. Pas de couverture exhaustive — viser les pièges (frontières, NaN, fuseau horaire).
- **Linting** : `ruff check` + `ruff format` (rapide, remplace `black` + `flake8` + `isort`).
- **Type hints** sur toutes les signatures publiques, vérifiés par `mypy --strict` sur `src/`.
- **Pre-commit hook** (optionnel) qui lance ruff + mypy avant chaque commit.

---

## 6. Reproductibilité

- `pyproject.toml` + `uv.lock` versionnés.
- `SEED = 42` propagé partout (numpy, sklearn, sampling background).
- Manifestes de données : `data/raw/MANIFEST.md` liste URLs + checksums SHA256 des téléchargements.
- Rapport Quarto régénérable par `quarto render rapport/rapport.qmd`.

---

## 7. Pièges spécifiques Python à anticiper

1. **Polars vs pandas** : l'EBD doit être lu en `polars.scan_csv` (lazy) ; convertir en pandas uniquement après le filtre. Sinon RAM saturée.
2. **`rioxarray.reproject_match` chunked** : sur les rasters LiDAR mosaïqués, activer `rasterio.Env(GDAL_CACHEMAX=512)` ou passer par `dask`. Sinon OOM.
3. **`elapid` n'a pas de tuning intégré** : implémenter la grille AICc à la main. Référence : Muscarella et al. 2014 (ENMeval).
4. **`BalancedRandomForestClassifier`** rééchantillonne à chaque arbre, pas globalement → ne pas down-sampler en amont.
5. **`pylandstats` attend un raster catégoriel int** : caster avant focal_metric.
6. **CRS implicite** : `xarray` perd le CRS au moindre `.where()` ou arithmétique. Toujours `da.rio.write_crs(CRS_TRAVAIL, inplace=True)` après transformation.
7. **`astral` veut un fuseau horaire** : `pytz.timezone("America/Toronto")` + DST géré ; ne pas mélanger UTC et local naïvement.

---

## 8. Livrables Python additionnels (vs version R)

Identique à la liste S5 du plan R, plus :

- Dépôt Git avec `pyproject.toml` + `uv.lock` (au lieu de `renv.lock`).
- `README.md` avec commande `uv sync && python scripts/run_all.py`.
- Modèles sérialisés (`joblib` pour RF, dict de params pour MaxEnt).

---

*Document de protocole v1.0 — à raffiner après la prise en main de `elapid` et le test du filtrage EBD sur une vraie MRC en S1.*
