# Revue critique du protocole Python — note de direction

**De :** direction de projet (modélisation géospatiale / DL appliqué)
**À :** A. Chéné
**Objet :** revue de `04_protocole_python.md` — simplifier avant d'implémenter
**Date :** 2026-06-11

---

## Synthèse

Le protocole est **techniquement défendable mais surdimensionné pour un POC de 4–5 semaines**. Tu as transposé la stack R en Python en gardant la même ambition (deux modèles, validation lourde, bootstrap, MESS, métriques exotiques). Conséquence : six dépendances tierces qui peuvent être remplacées par 50 lignes de NumPy/scikit-learn, deux modèles à valider/interpréter là où un suffit, et un risque réel d'enlisement en semaine 2 sur le prétraitement de 80 GB de raster. Avant d'écrire une ligne de code modélisation, on simplifie.

L'essence du projet, qu'il faut garder en tête à chaque décision :

> *Cartographier la probabilité d'occurrence de l'Engoulevent sur une MRC, à partir d'observations eBird et de variables environnementales, et identifier des zones favorables non échantillonnées.*

Tout ce qui ne sert pas directement cette phrase est candidat à la coupe.

---

## Problème 1 — Le volume raster (80 GB / 107 fichiers) est ton vrai bottleneck

C'est le point qui me préoccupe le plus, et qui n'est pas adressé dans le protocole. À 750 MB/fichier en moyenne (LiDAR 1 m), trois pièges t'attendent :

1. **`rioxarray.reproject_match` sur la mosaïque complète va exploser la RAM.** Tu n'as pas 80 GB de RAM. La fonction charge tout en mémoire avant de rééchantillonner.
2. **Mosaïquer puis rééchantillonner = pire stratégie.** Tu écris un fichier intermédiaire de ~80 GB pour produire un raster de 500 MB.
3. **Lire 107 fichiers à la volée à chaque étape** (extraction, focal, plot) = I/O catastrophique.

### Solution : pyramide inversée — réduire AVANT de mosaïquer

```
107 fichiers × 750 MB (1 m)
        │
        ▼  pour CHAQUE tuile, indépendamment, en parallèle :
           rasterio.open → fenêtre par fenêtre → agrégation à 250 m
           → écrire petit COG (~5 MB)
        │
        ▼
107 fichiers × 5 MB (250 m) = ~500 MB total
        │
        ▼  une seule opération de mosaïque (gdal.BuildVRT ou rio merge)
        │
        ▼
1 raster MHC_250m.tif aligné sur la grille MRC
```

**Code minimum requis** : `rasterio.open(tile)` → `Resampling.average` via `read(out_shape=...)` → `rasterio.write` en COG. Une fonction de ~25 lignes, parallélisée via `concurrent.futures.ProcessPoolExecutor` (8–12 workers selon ton CPU). **Estimation : 20–40 min pour 107 tuiles sur un MacBook.**

Bénéfice : tout le reste du pipeline travaille sur des rasters de 500 MB. Plus de problème de mémoire, plus besoin de `dask`, plus besoin de chunking exotique dans `rioxarray`.

> **Recommandation forte :** code cette étape en semaine 1, immédiatement après le téléchargement. Tant que tu n'as pas les MHC/TWI/Pentes à 250 m sur disque, tu ne peux rien faire de productif.

---

## Problème 2 — Deux modèles pour un POC

Le plan R prévoyait MaxEnt **et** Random Forest. En Python, tu hérites de cette ambition, mais elle pèse plus lourd : `elapid` est moins mature que `maxnet`, son tuning n'a pas d'équivalent ENMeval, et tu devras coder la grille AICc à la main.

**Question directe : qu'apporte MaxEnt que RF ne te donne pas ?**

Réponse honnête pour un POC : pas grand-chose. La comparaison méthodologique est intéressante en thèse, pas en POC de 4 semaines. Choisis **un modèle, bien fait** plutôt que deux à moitié.

Mon vote : **Random Forest sur checklists zero-fillées**, parce que :

- Tu as 2 500+ présences ET des absences crédibles (listes complètes filtrées Johnston). C'est exactement le régime de données où RF brille. MaxEnt est conçu pour le cas pathologique inverse (présence-seule).
- Aucune dépendance exotique : `scikit-learn` suffit.
- Interprétabilité : permutation importance + partial dependence plots dans `sklearn.inspection`. Pas besoin de coder de courbes de réponse maison.
- La littérature eBird récente (Johnston et al. 2021, Strimas-Mackey et al. 2023) recommande explicitement cette approche.

Garde MaxEnt comme **paragraphe dans la discussion** ("approche alternative non retenue") plutôt que comme livrable. Tu gagnes une semaine.

---

## Problème 3 — Librairies à couper

| Lib proposée | Coupe ? | Remplacement | Économie |
|---|---|---|---|
| `polars` + `pandas` | garder une seule | `polars` partout (lazy scan EBD, conversion fin de pipeline si besoin sklearn) | clarté |
| `elapid` | **couper** | (drop MaxEnt) | 1 dépendance, 1 sem. de tuning |
| `pylandstats` | **couper** | `scipy.ndimage.uniform_filter` pour densité de lisière (binarise forêt/non-forêt, `sobel` ou diff finie, focal sum) — 15 lignes | 1 dépendance |
| `verde.BlockKFold` | **couper** | splitter spatial maison : `np.floor(coords / taille_bloc)` → groupes → `GroupKFold` sklearn. ~20 lignes | 1 dépendance |
| `skgstat` (variogramme) | **couper** | choix *a priori* du bloc : 5–10 km, basé sur home-range de l'espèce (Wilson & Watts 2008 ≈ 100 ha → 1 km de rayon × 5 par sécurité). Le variogramme des résidus est un raffinement de thèse. | 1 dépendance, 1 jour |
| `imbalanced-learn` | **couper** | `RandomForestClassifier(class_weight="balanced")` ; c'est presque équivalent à BRF et déjà dans sklearn | 1 dépendance |
| `astral` | garder | OK, compact et bien fait | — |
| `elapid` MESS | **couper la fonction** | MESS = ~30 lignes : pour chaque pixel, similarité min sur chaque var vs quantiles de l'entraînement. Code-le toi, c'est instructif. | — |
| `rioxarray` | garder | mais l'utiliser avec parcimonie (cf. problème 1) | — |
| `Quarto` | optionnel | `jupyter nbconvert` ou même Markdown + figures suffit pour un POC. Quarto est sympa si tu connais déjà. | flexibilité |

**Stack finale recommandée** : `numpy`, `pandas` (ou `polars`), `geopandas`, `rasterio`, `rioxarray`, `scikit-learn`, `matplotlib`, `astral`, `tqdm`. C'est tout. Neuf libs, toutes battle-tested, toutes en mémoire.

---

## Problème 4 — La granularité des scripts est trop fine

Douze scripts numérotés `01_` à `12_` pour 4–5 semaines, c'est de la cérémonie. Chaque script ajoute du boilerplate (argparse, logging, lecture/écriture de fichier intermédiaire). En pratique tu passeras autant de temps à orchestrer qu'à modéliser.

**Proposition : 5 scripts, structure plate, pas de package importable.**

```
projet_sdm_engoulevent/
├── pyproject.toml
├── README.md
├── data/                      # raw / interim / processed (déjà fait)
├── outputs/                   # figures / maps / models
├── code/
│   ├── utils.py               # I/O, CRS, helpers (~150 lignes)
│   ├── 01_predictors.py       # tile → 250m → mosaïque → stack final
│   ├── 02_ebird.py            # filter + zero-fill + extract covars
│   ├── 03_model.py            # RF + spatial CV + importance + PDP
│   ├── 04_predict.py          # carte proba + MESS + hotspots
│   └── 05_figures.py          # toutes les figures du rapport
├── notebooks/
│   └── exploration.ipynb      # un seul, pour les essais
└── rapport/
    └── rapport.md             # ou .qmd si tu y tiens
```

Pas de `src/sdm_engoulevent/` en package. Pas de `tests/`. Pas de mypy strict. Ces couches sont précieuses pour un code de production ; pour un projet de cours, elles coûtent du temps que tu mets mieux ailleurs (lecture de la littérature, qualité de la discussion).

**Règle** : `utils.py` contient ce qui est réutilisé ≥ 2 fois. Le reste vit dans le script qui l'utilise.

---

## Problème 5 — La validation est over-engineered

Le protocole prévoit :

- variogramme des résidus pour calibrer la taille de bloc,
- blockCV à 5 folds,
- AUC + TSS + CBI,
- bootstrap 100 réplicats pour la carte d'incertitude,
- MESS pour l'extrapolation.

C'est ce qu'on ferait pour un papier. Pour un POC :

- **Taille de bloc** : 10 km, fixée *a priori*, justifiée par l'écologie (home-range × marge). On rédige une phrase, on passe à autre chose.
- **AUC + TSS** : suffisent. CBI est intéressant mais coder + interpréter coûte 1–2 jours pour une métrique que personne dans ton jury ne lira en détail.
- **Bootstrap 100 réplicats × prédiction sur stack 250 m** : si la MRC fait 5 000 km², ça fait ~80 000 pixels × 100 modèles = 8 M prédictions. Faisable, mais l'incertitude RF est mieux estimée par la **variance entre les arbres** d'un seul modèle (`RandomForestClassifier` te donne `predict_proba` sur chaque arbre via `estimators_`). Une ligne, pas un bootstrap.
- **MESS** : garder, c'est peu cher (30 lignes) et c'est exactement la métrique qui répond à ta question (où est-ce que je peux extrapoler ? où est-ce que je dois rester prudent ?).

Tu gagnes facilement 4–5 jours.

---

## Sur le deep learning (mon domaine — et pourquoi je dis non ici)

Tu pourrais être tenté de te dire "tant qu'à passer en Python, pourquoi pas un CNN sur des patchs raster autour des présences ?" — l'approche de Deneu et al. (2021) en SDM par convolution. Réponse :

- **Volume de données** : 2 500 présences, c'est en bas du minimum viable pour un CNN à partir de zéro. Du *transfer learning* depuis ResNet pourrait marcher, mais ajoute 2 semaines.
- **Interprétabilité** : ton rapport doit défendre des hypothèses écologiques (H1–H4). Un CNN te donne une carte mais pas de courbes de réponse facilement interprétables. RF + permutation importance + PDP gagnent haut la main sur le critère pédagogique du cours.
- **Budget compute** : 80 GB de raster + entraînement d'un réseau = GPU souhaitable. Pas dans ton scope.

**Verdict** : RF est l'outil correct ici. Mentionne le CNN en discussion comme axe de prolongement, avec une ou deux références (Deneu 2021, Botella 2018). C'est ce qu'un jury appréciera.

---

## Inefficacités plus subtiles à corriger

1. **Background à 10 000 points pondéré par densité de checklists** : pertinent pour MaxEnt, **inutile pour RF zero-filled** (tu as déjà tes absences réelles via le zero-fill). Si tu drops MaxEnt, drop aussi le background.
2. **Extraction valeurs aux points par `xarray.sel`** : `rasterio.sample` est plus rapide d'un facteur ~10 sur des points épars. Préfère-le.
3. **Stack multi-bandes en GeoTIFF** : OK, mais nomme-les avec `set_band_description` pour ne pas perdre l'ordre. `xarray.Dataset` + netCDF est plus robuste.
4. **VIF + corrélation Spearman + élimination itérative** : trois étapes redondantes. Pour 6 variables, fais une matrice de corrélation, regarde-la, justifie tes choix en une phrase. Pas besoin d'algorithme automatique.
5. **`SEED = 42` propagé partout** : oui, mais attention — sklearn refit les estimateurs avec un seed différent par défaut. Passe `random_state=SEED` partout (RF, splitters, sampling).
6. **Quarto avec kernel Python** : le rendu est sympa mais le debugging est pénible. Pour un rapport de cours, Markdown + figures incluses suffit. Garde Quarto si tu y tiens vraiment.

---

## Structure de pipeline recommandée (version simplifiée)

```
01_predictors.py
    Pour chaque tuile (107×) en parallèle :
        lire 1 m, agréger à 250 m (Resampling.average), écrire COG
    Construire VRT → mosaïque MHC / TWI / Pentes à 250 m
    Clip CHELSA Bio10 sur l'extent MRC + reproject à 250 m
    Rasteriser carte écoforestière → classes forêt/ouvert/...
    Calculer densité de lisière par scipy.ndimage focal (fenêtre 1 km)
    Calculer densité de routes (rasterize lignes + focal sum)
    Empiler 6 bandes alignées → stack_250m.nc
    [Temps cible : 1 jour]

02_ebird.py
    polars.scan_csv EBD + sampling (lazy)
    Filtres espèce / dates / protocoles / listes complètes / Johnston
    Zero-fill (left-join sur SAMPLING_EVENT_IDENTIFIER)
    Calcul variables détection (durée, minutes-après-coucher, lune)
    Extraction valeurs stack aux points (rasterio.sample)
    → table_modele.parquet
    [Temps cible : 1 jour]

03_model.py
    Split spatial : np.floor(coord / 10000) → group_id → GroupKFold(5)
    RandomForestClassifier(class_weight="balanced", random_state=SEED)
    RandomizedSearchCV sur n_estimators, max_features, min_samples_leaf
        scoring="roc_auc", cv=GroupKFold
    Refit sur tout
    Permutation importance + partial dependence (sklearn.inspection)
    → rf_model.joblib + importance.csv + pdp_plots/
    [Temps cible : 1 jour de calcul + 2 jours d'analyse]

04_predict.py
    Charger stack, reshape (H*W, 6), supprimer NaN
    rf.predict_proba(X) par chunks de 1M pixels
    Variance entre arbres pour incertitude (estimators_)
    MESS maison (30 lignes)
    Identifier top-5 hotspots (proba > 0.7) × faible effort eBird
    → proba.tif, incertitude.tif, mess.tif, hotspots.gpkg
    [Temps cible : 1 jour]

05_figures.py
    Toutes les figures du rapport, paramétrées pour la sortie finale.
    [Temps cible : 2 jours, en parallèle de la rédaction]
```

**Planning révisé : S1 cadrage + zone d'étude + reduce LiDAR ; S2 ebird + stack final ; S3 modélisation + validation ; S4 prédiction + figures ; S5 rédaction. Tu gardes ton buffer S6.**

---

## Questions à te poser avant de coder

1. **Ai-je déjà téléchargé les 80 GB de LiDAR ?** Si non, c'est ta priorité absolue. Le reste peut commencer en parallèle dès que tu as ne serait-ce que la moitié des tuiles.
2. **Mon home directory a-t-il 200 GB libres ?** (raw + interim + processed + outputs)
3. **Est-ce que je sais déjà qui sera mon évaluateur et qu'est-ce qu'il/elle valorise ?** Méthodologie rigoureuse > nombre de méthodes comparées, généralement.
4. **Le rapport** : 15–20 pages dont la moitié sera de la discussion. La modélisation est un moyen, pas la fin.

---

## Ce que je te demande de livrer cette semaine

Pas du code de modèle. Pas un environnement Python parfait. Juste :

1. **Confirmation de la MRC retenue** (S1 du plan original).
2. **Le script de réduction tile → 250 m** fonctionnel sur 3–5 tuiles tests, mesuré (temps, mémoire pic, taille de sortie). Si ça marche, on extrapole sur les 107 tuiles en confiance.
3. **Une décision écrite** : on garde RF seul, ou on insiste sur la comparaison MaxEnt/RF (avec la semaine de retard que ça implique) ? Réponds-moi en 3 lignes.

Le reste se construit sur ces fondations.

---

*Note de revue v1.0 — à amender après réponse de l'étudiant sur les 3 points ci-dessus.*
