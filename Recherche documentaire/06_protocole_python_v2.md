# GMQ405 — Protocole méthodologique Python v2
## SDM Engoulevent bois-pourri — version simplifiée et rééchelonnée à 5 m

**Version :** 2.0 — 2026-06-11
**Remplace :** `04_protocole_python.md` (v1)
**Décisions structurantes :**
- résolution de travail **5 m** (10 m en repli si la prédiction sort de l'enveloppe mémoire) ;
- **un seul modèle** : Random Forest sur checklists zero-fillées ;
- stack minimale de librairies, scripts plats, validation pragmatique.

---

## 1. Cadre

Question, hypothèses (H1–H4) et variables (MHC, TWI, lisière, prop. feuillu/mélangé, densité routes, Bio10) inchangées par rapport au plan `03_plan_projet_session.md`. Le présent document fige uniquement *comment* on les produit.

Implication directe du passage à 5 m : on traite 80 GB de LiDAR en entrée, on produit une stack de prédicteurs à 5 m de ~3 GB, et on prédit sur ~10⁸ pixels. Tout le pipeline doit fonctionner par **fenêtre/tuile**, jamais sur la mosaïque complète en RAM.

---

## 2. Stack technique

Neuf librairies, toutes en mémoire courante. Rien d'exotique.

| Usage | Lib | Notes |
|---|---|---|
| Tabulaire | `polars` | `scan_csv` lazy pour l'EBD ; conversion `to_pandas()` seulement à l'entrée du modèle |
| Vecteur | `geopandas` | clip, reprojection, spatial join |
| Raster | `rasterio` + `rioxarray` | `rasterio` pour les opérations fenêtrées et la mosaïque VRT ; `rioxarray` uniquement quand la stack est déjà petite (lecture finale, plot) |
| Numérique | `numpy` + `scipy.ndimage` | focal stats (uniform_filter, generic_filter) — remplace `pylandstats` |
| Modèle | `scikit-learn` | RF avec `class_weight="balanced"` ; permutation importance ; partial dependence |
| Soleil/lune | `astral` | minutes après coucher du soleil, fraction lunaire |
| Plots/cartes | `matplotlib` + `contextily` | basemap OSM pour les cartes finales |
| Progression | `tqdm` | barres pour les boucles tuile par tuile |
| Repro | `uv` + `pyproject.toml` | lockfile versionné |

**Coupé volontairement** : `elapid`, `pylandstats`, `verde`, `skgstat`, `imbalanced-learn`, `pandas` (sauf à l'entrée sklearn), `Quarto` (Markdown + figures suffit), `dask` (la stratégie tile-by-tile l'évite).

---

## 3. Arborescence

Structure plate, pas de package, pas de tests unitaires. Les modules partagés vivent dans `utils.py`.

```
projet_sdm_engoulevent/
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore                # exclut data/raw, outputs/maps
│
├── data/
│   ├── raw/                  # téléchargements (déjà existant : MNT/, MHC/, Pentes/, chelsa/, ...)
│   ├── interim/              # tuiles agrégées à 5 m
│   │   ├── MHC_5m/           # 107 petits COGs (~50 MB chacun)
│   │   ├── TWI_5m/
│   │   └── Pentes_5m/
│   └── processed/
│       ├── stack_5m.vrt      # mosaïque virtuelle des 6 bandes alignées
│       ├── checklists.parquet
│       └── table_modele.parquet
│
├── code/
│   ├── utils.py              # I/O, CRS, helpers, focal_density, mess
│   ├── 01_predictors.py      # tuile → 5 m → VRT mosaïque + variables paysagères
│   ├── 02_ebird.py           # filter + zero-fill + extract covars
│   ├── 03_model.py           # RF + spatial CV + importance + PDP
│   ├── 04_predict.py         # carte proba (windowed) + MESS + hotspots
│   └── 05_figures.py         # figures du rapport
│
├── notebooks/
│   └── exploration.ipynb     # un seul, pour les essais
│
├── outputs/
│   ├── figures/
│   ├── maps/                 # GeoTIFF COG + PNG
│   ├── tables/
│   └── models/               # rf.joblib + métadonnées
│
└── rapport/
    └── rapport.md            # + figures référencées
```

---

## 4. Stratégie pour le 5 m sur 80 GB

Le principe : **rien ne charge la mosaïque complète en RAM, jamais**. Trois mécanismes combinés.

### 4.1 Réduction tuile par tuile

Chaque tuile LiDAR brute (~750 MB à 1 m) est lue par fenêtres, agrégée à 5 m via moyenne (variables continues) ou mode (catégorielles si jamais), et écrite en COG. La lecture/écriture fenêtrée garantit une empreinte mémoire constante (~200 MB) quel que soit le nombre de tuiles.

Bilan attendu : 80 GB (1 m) → ~3 GB (5 m) répartis sur 107 petits COGs. Parallélisable sur 8–12 workers via `ProcessPoolExecutor`.

### 4.2 Mosaïque virtuelle (VRT)

On ne mosaïque jamais physiquement les 107 fichiers. À la place, `gdal.BuildVRT` produit un `.vrt` (XML de quelques KB) qui pointe vers les tuiles. Lecture transparente comme un seul raster. Idéal pour `rasterio.windows.from_bounds` et la prédiction fenêtrée.

### 4.3 Prédiction fenêtrée

Pour la carte finale, on itère sur des fenêtres de ~2 048 × 2 048 pixels (~16 MB par bande), on lit le stack, on flatten en (N, 6), on appelle `rf.predict_proba`, on reshape, on écrit dans le COG de sortie. Empreinte RAM ~500 MB en permanence, même sur une MRC de 5 000 km² (200 M pixels).

---

## 5. Étapes opérationnelles

### Script 01 — `01_predictors.py`

Construit la stack à 5 m, alignée sur la grille MRC.

**Fonctions principales (dans `utils.py` ou inline) :**

- `agreger_tuile_5m(path_in, path_out, facteur=5, methode="average") -> None` — lecture fenêtrée d'une tuile 1 m, agrégation à 5 m, écriture COG. Empreinte RAM bornée.
- `mosaiquer_vrt(tuiles_5m, path_vrt, bounds=None) -> None` — `gdal.BuildVRT` sur les COGs ; `bounds` optionnel pour clipper à la MRC.
- `rasteriser_polygones(gdf, grille_template, attr=None, fill=0) -> np.ndarray` — `rasterio.features.rasterize` ; sert pour écoforestière → forêt/non-forêt, et pour routes (longueur ligne par pixel).
- `densite_lisiere(raster_binaire, fenetre_m, resolution_m) -> np.ndarray` — détecte les pixels de bordure (différence finie sur 3×3), puis `scipy.ndimage.uniform_filter` de taille `fenetre_m / resolution_m` pour la densité focale. Sortie en m de lisière / ha. Remplace `pylandstats.edge_density`.
- `proportion_focale(raster_binaire, fenetre_m, resolution_m) -> np.ndarray` — `uniform_filter` direct.
- `densite_routes(routes_gdf, grille_template, fenetre_m) -> np.ndarray` — rasterise longueur de ligne par pixel, focal sum, divise par aire focale → km/km².
- `clip_chelsa(path_chelsa, mrc_gdf, grille_5m) -> np.ndarray` — clip Bio10 + `reproject` en bilinéaire vers la grille 5 m. C'est la seule variable qui passera par un upsampling massif (1 km → 5 m) — c'est attendu et écologiquement correct (climat varie lentement).

**Pipeline du script :**

1. Lire le polygone MRC, définir la grille de référence à 5 m (un dummy raster sert de template).
2. En parallèle (8–12 workers), agréger les 107 tuiles MNT/MHC/Pentes à 5 m → `data/interim/<produit>_5m/`.
3. Calculer TWI à partir des MNT 5 m mosaïqués (équation classique : `ln(a / tan(β))` où `a` = aire amont par cellule, `β` = pente locale). Pour un POC, prendre le TWI **déjà fourni par Données Québec** si disponible, sinon `pysheds` est l'unique exception à autoriser ici.
4. `BuildVRT` pour chaque produit, clip sur l'extent MRC.
5. Rasterise carte écoforestière → binaire forêt → `densite_lisiere` (1 km) et `proportion_focale` feuillu/mélangé (500 m).
6. Rasterise routes → `densite_routes` (1 km).
7. `clip_chelsa` Bio10.
8. Assembler **6 bandes** dans un VRT multi-bandes ou un GeoTIFF multi-bandes COG : `stack_5m.tif`.

**Livrable** : `data/processed/stack_5m.tif` (multi-bandes, ~3 GB compressé DEFLATE, lecture fenêtrée).

**Temps cible** : 1.5 jour avec parallélisation.

---

### Script 02 — `02_ebird.py`

Construit la table modélisation.

**Fonctions :**

- `charger_ebd(path_obs, path_sampling) -> tuple[pl.LazyFrame, pl.LazyFrame]` — `polars.scan_csv` lazy.
- `filtrer_ebird(lf_obs, lf_sampling, mrc_bounds, espece="Eastern Whip-poor-will") -> pl.LazyFrame` — chaîne unique : espèce, dates juin–juillet, protocoles Stationary/Traveling, listes complètes, filtres Johnston 2021 (durée ≤ 300 min, distance ≤ 5 km, ≤ 10 obs), bounding-box de la MRC.
- `zero_fill(obs, sampling, espece) -> pl.DataFrame` — left-join sur `SAMPLING EVENT IDENTIFIER`, colonne `presence` binaire.
- `variables_detection(df) -> pl.DataFrame` — ajoute : `minutes_apres_coucher` (via `astral`, fuseau `America/Toronto`, DST géré), `phase_lune` (fraction éclairée), `jour_julien`, log de `DURATION_MINUTES`.
- `extraire_stack(df, stack_path, bandes) -> pl.DataFrame` — `rasterio.sample` sur les points (10× plus rapide que `xarray.sel` sur des points épars). Joint les 6 valeurs au DataFrame.

**Pipeline :**

1. Charger EBD + sampling (lazy).
2. Filtrer + zero-fill → ~10⁴–10⁵ checklists dont quelques milliers de présences.
3. Calculer variables de détection.
4. Extraire les 6 prédicteurs aux coordonnées.
5. Supprimer les lignes avec NaN (bords de MRC).
6. Écrire `data/processed/table_modele.parquet`.

**Livrable** : table prête à modéliser, colonnes : `lat, lon, date, presence, MHC, TWI, lisiere, prop_feuillu, dens_routes, bio10, duree, min_post_coucher, phase_lune, jour_julien`.

**Temps cible** : 1 jour.

---

### Script 03 — `03_model.py`

Random Forest avec validation spatiale.

**Fonctions :**

- `bloc_spatial_groupes(df, taille_bloc_m=10_000, crs_proj="EPSG:32198") -> np.ndarray` — `np.floor(coords_x / taille) * 10_000 + np.floor(coords_y / taille)` → identifiant de bloc par observation. Sert de `groups` pour `GroupKFold`.
- `tss(y_true, y_proba, seuil=0.5) -> float` — sensibilité + spécificité − 1. Quatre lignes.
- `evaluer_cv(model, X, y, groups, k=5) -> pd.DataFrame` — `GroupKFold(k)`, retourne AUC + TSS par fold.

**Pipeline :**

1. Charger `table_modele.parquet`, séparer `X` (occupation + détection) et `y` (presence).
2. Calculer `groups` via `bloc_spatial_groupes` (10 km, justifié par home-range Engoulevent ≈ 100 ha × marge ×100).
3. **Tuning** : `RandomizedSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42), {n_estimators: [500, 1000], max_features: ["sqrt", 0.33], min_samples_leaf: [1, 5, 10], max_depth: [None, 20, 40]}, cv=GroupKFold(5), scoring="roc_auc", n_iter=20)`.
4. **Validation** : `evaluer_cv` sur le meilleur modèle, 5 folds spatiaux.
5. **Refit** sur tout, sérialiser `joblib.dump(rf, "outputs/models/rf.joblib")`.
6. **Interprétation** :
   - `sklearn.inspection.permutation_importance(rf, X, y, n_repeats=10, random_state=42)` — barplot des importances.
   - `sklearn.inspection.PartialDependenceDisplay.from_estimator(rf, X, ["MHC", "TWI", "lisiere", "prop_feuillu", "dens_routes", "bio10"])` — 6 courbes de réponse, une commande.

**Livrables** : `rf.joblib`, `cv_metriques.csv` (AUC + TSS par fold), `importance.csv`, `pdp_*.png`.

**Temps cible** : 1 jour de calcul/tuning + 1–2 jours d'analyse écologique des résultats.

---

### Script 04 — `04_predict.py`

Cartographie prédictive + incertitude + MESS + hotspots.

**Fonctions :**

- `predire_fenetre(rf, src_stack, dst_proba, dst_var, fenetre_size=2048) -> None` — itère sur les fenêtres du stack, applique `rf.predict_proba`, écrit dans deux COGs (probabilité moyenne + variance entre arbres).
- `variance_entre_arbres(rf, X) -> np.ndarray` — `np.var([tree.predict_proba(X)[:, 1] for tree in rf.estimators_], axis=0)`. Incertitude gratuite, pas de bootstrap nécessaire.
- `mess(X_train_quantiles, X_predict) -> np.ndarray` — pour chaque pixel, pour chaque variable : `100 * (min(d_min, d_max) / range)` où `d_min/max` = distance aux quantiles 0/100 de l'entraînement. Retourne le minimum sur les variables. Implémentation : ~30 lignes, vectorisée NumPy.
- `identifier_hotspots(proba_path, effort_raster, seuil_proba=0.7, seuil_effort_pct=20, top_n=5) -> gpd.GeoDataFrame` — masque proba > 0.7 ∧ effort < percentile 20 ; `scipy.ndimage.label` pour clusteriser ; retourne les 5 plus grandes zones avec centroïde, surface, proba moyenne.

**Pipeline :**

1. Prédire la carte de probabilité (fenêtré) → `outputs/maps/proba_rf.tif` (COG).
2. Variance entre arbres → `outputs/maps/incertitude.tif`.
3. MESS sur la stack vs distribution d'entraînement → `outputs/maps/mess.tif`.
4. Hotspots non-échantillonnés → `outputs/maps/hotspots.gpkg` + table de coordonnées.

**Livrables** : 3 GeoTIFF COG + 1 GeoPackage de polygones.

**Temps cible** : 1 jour (la prédiction fenêtrée tourne ~30–60 min sur une MRC moyenne).

---

### Script 05 — `05_figures.py`

Toutes les figures du rapport, paramétrées pour la sortie finale.

- Courbes de réponse partielles (6, en grille 2×3).
- Importance permutation (barplot horizontal).
- Boxplots AUC + TSS sur les 5 folds.
- Carte de probabilité MaxEnt-style avec basemap `contextily`.
- Carte d'incertitude (échelle inversée pour lisibilité).
- Carte MESS (seuil 0 : intérieur vs extrapolation).
- Carte des 5 hotspots annotés.

Une figure = une fonction, un appel. Pas de helper inutile.

**Temps cible** : 2 jours, en parallèle de la rédaction.

---

## 6. Validation — choix justifiés

- **Taille de bloc** : 10 km, *a priori*. Justification écologique : home-range Engoulevent ~100 ha (Wilson & Watts 2008) → rayon ~600 m → 10 km découple largement deux blocs voisins. Variogramme des résidus mentionné en discussion comme raffinement non-réalisé.
- **Métriques** : AUC-ROC + TSS, 5 folds spatiaux. Pas de CBI : intéressant en thèse, non-discriminant en POC pour le coût de mise en œuvre.
- **Incertitude** : variance entre arbres du RF (`estimators_`). Pas de bootstrap.
- **Extrapolation** : MESS, seul outil de cette liste qui sert directement la discussion ("où ne pas faire confiance au modèle ?").

---

## 7. Planning révisé (5 semaines + 1 buffer)

| Semaine | Tâche | Livrable |
|---|---|---|
| S1 | Confirmer MRC, vérifier couverture LiDAR, tester `agreger_tuile_5m` sur 3 tuiles, mesurer (temps, RAM, taille de sortie) | Note de validation + script tile-by-tile prouvé |
| S2 | Téléchargement complet (si pas fait), réduction des 107 tuiles, construction stack 5 m, EBD filtré, table_modele | `stack_5m.tif` + `table_modele.parquet` |
| S3 | Modélisation RF, tuning, validation spatiale, interprétation écologique | `rf.joblib`, importances, PDP, métriques CV |
| S4 | Prédiction fenêtrée, incertitude, MESS, hotspots, premières figures | Cartes finales + GeoPackage hotspots |
| S5 | Rédaction rapport, figures finales, présentation orale | Rapport Markdown + dépôt Git |
| S6 | Buffer | — |

---

## 8. Pièges spécifiques à la résolution 5 m

1. **GPS eBird** : précision typique 5–30 m. À 5 m, un point eBird peut tomber sur le mauvais pixel. **Mitigation** : extraire la moyenne dans un buffer de 30 m autour du point (3×3 pixels), pas la valeur du pixel exact. Une option à ajouter dans `extraire_stack`.
2. **Bio10 à 5 m est un upsampling massif** (1 km → 5 m = ×200). C'est attendu mais documente-le. Le climat ne porte pas d'info à 5 m ; c'est juste une covariable à grande échelle qu'on uniformise par contrainte de stack.
3. **Focal 1 km à 5 m = noyau 200×200** : `scipy.ndimage.uniform_filter` reste séparable et rapide (~1–2 min sur la MRC entière), mais pas naïvement implémenté à la main. Toujours `uniform_filter`, jamais `generic_filter` qui n'est pas séparé.
4. **`rasterio` + multi-process + GDAL_CACHEMAX** : chaque worker doit avoir son cache GDAL borné (`os.environ["GDAL_CACHEMAX"] = "512"`) sinon ils se battent pour la RAM.
5. **Prédiction sur ~10⁸ pixels** : `rf.predict_proba` peut être lent ; testes sur une fenêtre puis extrapole. Si > 1 h sur la MRC complète, repli à 10 m (×4 plus rapide).
6. **Écriture COG** : `rasterio.open(path, "w", driver="COG", compress="DEFLATE", blocksize=512)`. Sinon les COGs ne sont pas optimisés pour la lecture fenêtrée ultérieure.

---

## 9. Pièges déjà identifiés en v1 et qui restent valables

- CRS : `rio.write_crs(EPSG:32198, inplace=True)` après chaque opération qui pourrait le perdre.
- Astral : timezone explicite, DST géré.
- Polars : convertir en pandas/numpy seulement à l'entrée sklearn.

---

## 10. Livrables finaux

1. Rapport Markdown ~15–20 pages : intro, méthodes, résultats, discussion, limites.
2. Dépôt Git public reproductible : `git clone && uv sync && python code/01_predictors.py ... && ... && python code/05_figures.py`.
3. Cartes : probabilité RF, incertitude, MESS, hotspots — toutes en COG + PNG.
4. Table des 5 hotspots prédits non-échantillonnés (coordonnées + caractéristiques environnementales).
5. Présentation orale si requise.

---

## 11. Décisions ouvertes (à trancher en S1)

- **MRC retenue** : reste à confirmer sur la base de `01_select_mrc` (cf. plan original).
- **TWI source** : LiDAR Données Québec en direct si disponible pour la MRC, sinon recalcul via `pysheds` (seule exception aux 9 libs).
- **Si la prédiction 5 m dépasse 2 h** : repli sur 10 m (changement de constante dans `utils.py`, le reste du code est invariant).

---

*Protocole v2.0 — concis, ciblé, ré-implémentable en 5 semaines. Toute déviation devra être justifiée par un gain mesurable sur la question de recherche, pas par appétit méthodologique.*
