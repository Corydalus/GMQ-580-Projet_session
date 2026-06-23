# Méthodologie — SDM Engoulevent bois-pourri
## Document source pour la génération du README

> Ce fichier est un brief complet destiné à Claude Code.
> Il synthétise le plan de projet, le protocole technique v2 et l'analyse des variables retenues.
> Claude Code doit lire ce fichier, puis lire le README existant du repo pour en respecter la structure imposée par le professeur, et enfin rédiger le README final en combinant les deux.

---

## 1. Informations générales

| Champ | Valeur |
|---|---|
| Cours | GMQ-580 — Géoinformatique II |
| Session | Session 3 |
| Auteur | Alexandre Chéné |
| Email | alex.chene05@gmail.com |
| Repo GitHub | https://github.com/Corydalus/GMQ-580-Projet_session |
| Espèce cible | *Antrostomus vociferus* — Engoulevent bois-pourri |
| Zone d'étude | zone_etude.gpkg |
| Statut | Preuve de concept (POC) — 5 semaines + 1 semaine tampon |
| Langue du rapport | Français |

---

## 2. Question de recherche

> *Quelles variables environnementales structurent la présence de l'Engoulevent bois-pourri à l'échelle d'une MRC du Québec méridional, et où le modèle prédit-il des habitats favorables non encore documentés par eBird ?*

---

## 3. Hypothèses écologiques

| # | Hypothèse |
|---|---|
| H1 | La probabilité d'occurrence augmente avec la densité de lisière forêt-ouvert jusqu'à un optimum intermédiaire (~40–60 % couvert forestier). |
| H2 | Les sols bien drainés (TWI bas) augmentent la probabilité de présence — préférence pour les substrats secs. |
| H3 | La densité de routes a un effet non-linéaire : faiblement positif à faible densité (lisière + insectes), négatif au-delà d'un seuil (dérangement). |
| H4 | Les peuplements de hauteur et d'âge intermédiaires (jeunes et mi-âges, ~10–20 m) sont préférés aux peuplements matures fermés. |

---

## 4. Variables retenues (10 variables finales)

### 4.1 Variables d'habitat (entrées du modèle RF)

| # | Variable | Source | Résolution native | Échelle d'analyse | Hypothèse testée |
|---|----------|--------|-------------------|-------------------|------------------|
| 1 | Hauteur moyenne de canopée (MHC) | LiDAR — Données Québec | 1 m → agrégé 5 m | 5 m + focal | H4 — structure verticale |
| 2 | Indice topographique d'humidité (TWI) | LiDAR — Données Québec | 1 m → agrégé 5 m | 5 m | H2 — drainage |
| 3 | Densité de lisière forêt-ouvert | Carte écoforestière MFFP | Vecteur → 5 m | Focal 1 km | H1 — mosaïque |
| 4 | Proportion forêt feuillue/mélangée | Carte écoforestière MFFP | Vecteur → 5 m | Focal 500 m | Composition |
| 5 | Densité de routes (km/km²) | Adresses Québec / RRN | Vecteur → 5 m | Focal 1 km | H3 — anthropique |
| 6 | Température estivale moyenne (Bio10) | CHELSA v2.1 | 1 km → resamp. 5 m | 1 km | Limite thermique |
| 7 | Classe d'âge du peuplement | Carte écoforestière MFFP | Vecteur → 5 m | 5 m | H4 — structure temporelle |
| 8 | Densité du peuplement (fermeture couvert) | Carte écoforestière MFFP | Vecteur → 5 m | 5 m | H4 — couvert complémentaire |
| 9 | Distance à un milieu humide | NHN / Canvec | Vecteur → 5 m | 5 m | Alimentation nocturne |
| 10 | Élévation (MNT) | LiDAR — Données Québec | 1 m → agrégé 5 m | 5 m | Limite altitudinale (~600 m) |

### 4.2 Variables de détection (RF uniquement — non cartographiées)

Ces variables corrigent le biais d'effort d'observation mais ne sont pas utilisées pour la prédiction spatiale.

| Variable | Calcul |
|---|---|
| `minutes_apres_coucher` | Minutes après le coucher du soleil local (via `astral`, fuseau `America/Toronto`, DST géré) |
| `phase_lune` | Fraction éclairée de la lune au moment de l'observation |
| `log_duree` | Log-transformée de la durée de la checklist (minutes) |
| `jour_julien` | Jour de l'année (centré sur la saison de reproduction juin–juillet) |

### 4.3 Variables exclues (avec justification)

| Variable | Raison d'exclusion |
|---|---|
| Domaine bioclimatique | Quasi-constant à l'échelle d'une MRC — variance nulle |
| Type de sol | Fortement corrélé avec TWI ; source SIGEOM complexe à intégrer |
| Peuplement écoforestier | Redondant avec type écologique + proportion feuillue |
| Type écologique | Catégoriel lourd (one-hot) ; corrélé avec prop. feuillue et densité peuplement |
| Années depuis perturbation | Corrélé avec classe d'âge (VIF > 5 attendu) |
| Type de perturbation | Faible priorité POC ; mentionner en discussion |
| NDVI | Prétraitement Sentinel-2 = ~1 semaine ; à intégrer dans une extension future |
| NDWI | Corrélé avec TWI et distance milieu humide |
| Pente | Déjà intégré implicitement dans TWI |
| Occupation du sol | Redondant avec type éco + prop. feuillue + densité lisière |
| Distance à cours d'eau | Redondant avec distance milieu humide |
| Ordre de Strahler | Calcul complexe, ROI faible pour un POC |
| Distance à un plan d'eau | Redondant avec distance milieu humide et cours d'eau |
| Distance à route | Remplacé par densité (plus informatif pour H3 non-linéaire) |
| Distance à milieu anthropique | Redondant avec densité routes |

---

## 5. Approche de modélisation

### Modèle unique : Random Forest (présence-absence)

- **Package** : `scikit-learn` — `RandomForestClassifier(class_weight="balanced", random_state=42)`
- **Données d'entrée** : zero-filling des checklists complètes eBird (`auk_zerofill`) — chaque checklist sans détection devient une absence confirmée
- **Filtres d'effort** (Johnston et al. 2021) :
  - Protocoles Stationary ou Traveling uniquement
  - Listes complètes (`ALL_SPECIES_REPORTED = 1`)
  - Durée ≤ 300 min, distance ≤ 5 km, ≤ 10 observateurs
  - Saison : juin–juillet (pic d'activité vocale de l'espèce)
- **Déséquilibre** : `class_weight="balanced"` — pas d'under-sampling manuel
- **Tuning** : `RandomizedSearchCV` — 20 itérations sur `n_estimators`, `max_features`, `min_samples_leaf`, `max_depth`
- **Validation** : `GroupKFold` avec blocs spatiaux de 10 km (découple les observations spatialement autocorrélées)
- **Métriques** : AUC-ROC + TSS sur 5 folds spatiaux
- **Interprétation** :
  - Importance par permutation (`sklearn.inspection.permutation_importance`)
  - Courbes de réponse partielle (PDP) pour les 10 variables d'habitat

### Pas de MaxEnt

La version antérieure du plan incluait MaxEnt. Ce modèle a été retiré pour concentrer l'effort sur un pipeline reproductible et bien validé. Le choix est justifiable : les listes complètes zero-fillées permettent un modèle présence-absence plus robuste que les méthodes présence-seule.

---

## 6. Pipeline technique

### Stack technologique

| Usage | Librairie | Notes |
|---|---|---|
| Tabulaire | `polars` | `scan_csv` lazy pour l'EBD ; `to_pandas()` seulement à l'entrée sklearn |
| Vecteur | `geopandas` | clip, reprojection, spatial join |
| Raster | `rasterio` + `rioxarray` | `rasterio` pour les opérations fenêtrées ; `rioxarray` pour les stacks légères |
| Numérique | `numpy` + `scipy.ndimage` | focal stats via `uniform_filter` (séparable = rapide) |
| Modèle | `scikit-learn` | RF + validation croisée + importance + PDP |
| Soleil/lune | `astral` | minutes après coucher du soleil, fraction lunaire |
| Cartes | `matplotlib` + `contextily` | basemap OSM pour les cartes finales |
| Progression | `tqdm` | barres pour les boucles tuile par tuile |
| Reproductibilité | `uv` + `pyproject.toml` | lockfile versionné |

**Gestion de la RAM** : résolution 5 m sur ~80 GB de LiDAR brut → tout le pipeline fonctionne par fenêtres/tuiles. Jamais de lecture de la mosaïque complète en RAM.

### Scripts (dans `code/`)

| Script | Rôle | Livrable |
|---|---|---|
| `utils.py` | Fonctions partagées : I/O, CRS, focal stats, MESS | — |
| `01_predictors.py` | LiDAR → tuiles 5 m → VRT → variables paysagères → `stack_5m.tif` | Stack 10 bandes, ~3 GB |
| `02_ebird.py` | EBD filtré → zero-fill → variables de détection → extraction covariables | `table_modele.parquet` |
| `03_model.py` | RF + tuning + validation spatiale + importance + PDP | `rf.joblib`, métriques CV |
| `04_predict.py` | Carte proba (fenêtrée) + incertitude + MESS + hotspots | GeoTIFF COG + GeoPackage |
| `05_figures.py` | Toutes les figures du rapport | PNG exports |

### Conventions techniques

- CRS : `EPSG:32198` (NAD83 / Québec Lambert) — unique pour tous les rasters et vecteurs
- Résolution : 5 m (repli 10 m si RAM insuffisante — changer `RESOLUTION_M` dans `utils.py`)
- COG : `driver="COG", compress="DEFLATE", blocksize=512` pour tous les rasters de sortie
- GDAL cache : `os.environ["GDAL_CACHEMAX"] = "512"` dans chaque worker multiprocessing
- Extraction eBird : buffer 30 m autour du point GPS (précision eBird ≈ 5–30 m)
- Reproductibilité : `random_state=42` partout, `uv.lock` versionné

---

## 7. Sources de données

| Donnée | Source | URL | Format | Taille approx. |
|---|---|---|---|---|
| eBird EBD (observations + sampling) | Cornell Lab of Ornithology | https://ebird.org/data/download | TSV compressé | 200–500 MB (filtré QC) |
| Carte écoforestière avec perturbations | Données Québec (MFFP) | https://www.donneesquebec.ca/recherche/dataset/carte-ecoforestiere-avec-perturbations | GDB / SHP | 50–200 MB |
| Produits dérivés LiDAR (MHC, MNT, Pentes) | Données Québec (MRNF) | https://www.donneesquebec.ca/recherche/dataset/produits-derives-de-base-du-lidar | GeoTIFF 1 m | 80 GB (~107 tuiles) |
| CHELSA Bio10 (température estivale) | CHELSA v2.1 | https://chelsa-climate.org/downloads/ | GeoTIFF 1 km | < 20 MB (clip MRC) |
| Réseau routier | Adresses Québec / RRN | https://www.donneesquebec.ca/recherche/dataset/adresses-quebec | SHP | < 50 MB |
| Réseau hydrographique national (milieux humides) | Ressources naturelles Canada | https://open.canada.ca/data/fr/dataset/a4b190fe-e090-4e6d-881e-b87956c07977 | GDB | 50–100 MB (MRC) |

**Budget données** : ~3–5 GB après prétraitement à 5 m. Faisable sur ordinateur portable (≥16 GB RAM recommandés).

**Note eBird** : les données EBD sont sous accord de confidentialité. Les fichiers bruts (`data/raw/ebird/`) sont exclus du repo Git via `.gitignore`. Voir les termes d'utilisation : https://www.birds.cornell.edu/home/ebird-data-access-terms-of-use/

---

## 8. Livrables

| # | Livrable | Format | Description |
|---|---|---|---|
| 1 | Rapport final | Markdown + figures | ~15–20 pages : intro, méthodes, résultats, discussion, limites |
| 2 | Dépôt Git reproductible | GitHub public | `git clone && uv sync && python code/01_predictors.py && ... && python code/05_figures.py` |
| 3 | Carte de probabilité RF | GeoTIFF COG + PNG | Probabilité de présence à 5 m sur la MRC |
| 4 | Carte d'incertitude | GeoTIFF COG + PNG | Variance entre les arbres du RF |
| 5 | Carte MESS | GeoTIFF COG + PNG | Zones d'extrapolation hors enveloppe d'entraînement |
| 6 | Hotspots non échantillonnés | GeoPackage + CSV | 5 zones à haute probabilité, faible effort eBird |

---

## 9. Planning

| Semaine | Tâches principales | Livrable |
|---|---|---|
| S1 | Confirmer MRC, vérifier couverture LiDAR, tester pipeline tuile sur 3 tuiles | Note de validation + repo initialisé |
| S2 | Réduction 107 tuiles → 5 m, stack 10 bandes, EBD filtré, table modèle | `stack_5m.tif` + `table_modele.parquet` |
| S3 | RF + tuning + validation spatiale + importance + PDP | `rf.joblib`, métriques CV, figures PDP |
| S4 | Prédiction fenêtrée, incertitude, MESS, hotspots | Cartes finales + GeoPackage |
| S5 | Rapport, figures finales | Rapport Markdown + dépôt Git finalisé |
| S6 | Buffer imprévus | — |

---

## 10. Références bibliographiques clés

- Johnston, A. et al. (2021). Analytical guidelines to increase the value of community science data: An example using eBird data to estimate species distributions. *Diversity and Distributions*, 27, 1265–1277. https://doi.org/10.1111/ddi.13271
- Strimas-Mackey, M. et al. (2023). *Best Practices for Using eBird Data*. Cornell Lab of Ornithology. https://ebird.github.io/ebird-best-practices/
- Guisan, A., Thuiller, W. & Zimmermann, N.E. (2017). *Habitat Suitability and Distribution Models: With Applications in R*. Cambridge University Press.
- Valavi, R. et al. (2019). blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods in Ecology and Evolution*, 10, 225–232.
- Wilson, M.D. & Watts, B.D. (2008). Response of Whip-poor-wills to landscape level habitat features in the Coastal Plain of Virginia. *Wilson Journal of Ornithology*, 120, 778–783.
- ECCC (2018). *Programme de rétablissement de l'Engoulevent bois-pourri au Canada*. Gouvernement du Canada.

---

## 11. Instructions pour Claude Code — génération du README

1. Lire ce fichier en entier.
2. Lire le README existant dans le repo (`cat README.md`) pour identifier :
   - La structure imposée par le professeur
   - Les sections obligatoires
   - Le modèle de README attendu
3. Générer le README final en **respectant exactement la structure du prof** et en remplissant chaque section avec le contenu de ce document.
4. Conserver les instructions du cours dans une section dédiée (`## Instructions du cours` ou équivalent).
5. Ne pas inventer de contenu — se limiter à ce qui est documenté ici.
6. Langue : **français**.
7. Ton : académique mais clair, adapté à un cours de géoinformatique de 2e cycle.

---

*Synthèse établie par Cowork (Claude) — juin 2026. Version 1.0.*
