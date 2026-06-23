# GMQ405 — Plan de projet de session
## SDM Engoulevent bois-pourri : modèle d'habitat et projection prédictive

**Cours :** GMQ405 — Modélisation et analyse spatiale
**Espèce :** *Antrostomus vociferus* (Engoulevent bois-pourri)
**Zone d'étude :** une MRC du Québec méridional (à définir ; sélection data-driven en semaine 1)
**Durée :** 4–5 semaines (buffer de 6 si requis)
**Statut :** *proof of concept* — effleure les concepts SDM + prédictif, pas un projet de maîtrise

---

## 1. Question de recherche

> *Quelles variables environnementales structurent la présence de l'Engoulevent bois-pourri à l'échelle d'une MRC du Québec méridional, et où le modèle prédit-il des habitats favorables non encore documentés par eBird ?*

**Objectifs opérationnels**

1. Ajuster deux modèles d'habitat sur les présences eBird (n ≥ 2 500) : un modèle présence-seule (MaxEnt) et un modèle présence-absence (Random Forest sur listes complètes zero-fillées).
2. Comparer leurs performances par validation spatiale croisée.
3. Produire une carte de probabilité d'occurrence sur l'ensemble de la MRC et identifier les zones favorables non échantillonnées.
4. Discuter les limites épistémiques (niche réalisée, biais d'effort, détection nocturne).

---

## 2. Hypothèses écologiques (à tester par le modèle)

- **H1** — La probabilité d'occurrence augmente avec la densité de lisière forêt-ouvert (mosaïque) jusqu'à un optimum intermédiaire (~40–60 % couvert forestier).
- **H2** — Les sols bien drainés (TWI bas) augmentent la probabilité de présence (préférence pour substrats secs).
- **H3** — La densité de routes a un effet non-linéaire : faiblement positif à faible densité (lisière + insectes), négatif au-delà d'un seuil (dérangement).
- **H4** — La hauteur intermédiaire de canopée (jeunes et mi-âges, ~10–20 m) est préférée aux peuplements matures fermés.

---

## 3. Variables retenues (jeu serré pour POC)

Coupe drastique par rapport au doc 02 : **6 variables d'occupation** + variables de détection (RF seulement). Règle: ≥10 présences par variable (avec 2 500 obs, on tient largement).

| # | Variable | Source | Résolution native | Échelle d'analyse | Justification |
|---|----------|--------|-------------------|-------------------|---------------|
| 1 | Hauteur moyenne canopée (MHC) | LiDAR Données Québec (dérivé) | 1 m → agrégé 250 m | 250 m, 1 km | H4 — structure verticale |
| 2 | Indice topographique d'humidité (TWI) | LiDAR Données Québec (dérivé) | 1 m → agrégé 250 m | 250 m | H2 — drainage |
| 3 | Densité de lisière forêt-ouvert | Carte écoforestière MFFP | vecteur → 250 m | 1 km | H1 — mosaïque |
| 4 | Proportion forêt feuillue/mélangée | Carte écoforestière MFFP | vecteur → 250 m | 500 m | composition |
| 5 | Densité de routes (km/km²) | Adresses Québec / RRN | vecteur → 250 m | 1 km | H3 — anthropique |
| 6 | Température estivale moy. (Bio10) | CHELSA v2.1 | 1 km | 1 km | covariable climatique stable |

**Variables de détection (RF/GLM seulement, exclues du MaxEnt)** : `DURATION_MINUTES`, `TIME_OBSERVATIONS_STARTED` (recentré sur coucher du soleil via `suncalc`), `NUMBER_OBSERVERS`, jour julien.

**Variables exclues volontairement** (et pourquoi) :
- VIIRS lumière nocturne : peu de variance à l'échelle d'une MRC rurale, ajoute peu pour beaucoup d'effort.
- NDVI Sentinel-2 : très intéressant mais ajoute 1 semaine de prétraitement.
- Pollution lumineuse, classe d'âge, dépôts de surface : Niveau 2/3 du doc 02, réservés à une extension.

---

## 4. Approches modélisées et comparaison

### 4.1 MaxEnt (présence-seule)
- Package : `maxnet` (pas de dépendance Java) ; tuning via `ENMeval`.
- Background : 10 000 points pseudo-absences pondérés par densité de checklists (bias file = densité de checklists eBird toutes espèces confondues, pour corriger le biais d'effort).
- Feature classes : linear + quadratic + hinge ; regularisation tunée par AICc.

### 4.2 Random Forest (présence-absence)
- Package : `ranger`.
- Données : zero-filling des checklists complètes (`auk_zerofill`) ; filtres Johnston 2021 (durée ≤ 5h, distance ≤ 5 km, ≤ 10 observateurs).
- Down-sampling de la classe majoritaire (RF balanced) pour gérer le déséquilibre présence/absence.

### 4.3 Comparaison
- Métriques : AUC-ROC, TSS, Boyce index continu (CBI).
- Validation : `blockCV` avec blocs de taille ≥ range du variogramme de la résiduelle ; 5 folds.
- Cartes prédictives sur la même grille (250 m) → différence par pixel + carte de concordance.
- Discussion : forces/faiblesses pédagogiques de chaque approche (interprétabilité, biais, suppositions).

---

## 5. Planning hebdomadaire

### Semaine 1 — Cadrage + acquisition + sélection MRC
- Mettre en place le projet R (`renv`, structure scripts numérotés, GitHub).
- Demander l'accès EBD (peut prendre quelques jours — démarrer dès J1).
- **Livrable S1** : script `01_select_mrc.R` qui compte les checklists Engoulevent par MRC, croise avec couverture LiDAR, propose 2–3 MRC candidates → décision finale.
- Acquisition : Carte écoforestière, dérivés LiDAR (MHC, TWI) pour la MRC retenue, CHELSA Bio10, réseau routier.
- **Livrable S1** : note de cadrage (1 page) + arborescence projet.

### Semaine 2 — Préparation des données
- Filtrage eBird avec `auk` (espèce, dates juin–juillet, protocoles Stationary/Traveling, listes complètes).
- Zero-filling, application des filtres d'effort.
- Reprojection de toutes les couches en NAD83 / Quebec Lambert (EPSG:32198), résolution commune 250 m, même extent.
- Calcul des variables à leurs échelles respectives (`terra::focal` ou `landscapemetrics`).
- Extraction valeurs aux points eBird, vérification VIF (< 5) et corrélations (< 0.7).
- **Livrable S2** : stack raster prêt + table presence/absence/covariables.

### Semaine 3 — Modélisation
- Ajustement MaxEnt avec tuning ENMeval (grille feature classes × regularisation).
- Ajustement Random Forest avec tuning `mtry` et `min.node.size`.
- Importance des variables (permutation pour les deux).
- Courbes de réponse partielles pour interprétation écologique.
- **Livrable S3** : 2 modèles ajustés + tableau d'importance + plots de réponses.

### Semaine 4 — Validation + cartographie prédictive
- Validation spatiale croisée `blockCV` (5 folds spatialement séparés).
- Calcul AUC, TSS, CBI sur folds tenus.
- Prédiction sur la pile raster → 2 cartes de probabilité (MaxEnt, RF).
- Carte MESS pour zones d'extrapolation hors domaine.
- Bootstrap (100 réplicats) pour carte d'incertitude.
- **Livrable S4** : cartes finales (probabilité MaxEnt, probabilité RF, désaccord, incertitude, MESS).

### Semaine 5 — Rédaction + figures + soutenance
- Rapport : introduction, méthodes, résultats, discussion, limites.
- Figures clés : courbes de réponse, validation, cartes.
- Identification des "hotspots" prédits non échantillonnés (top 5 polygones) → table de coordonnées + caractéristiques.
- **Livrable final** : rapport + dépôt Git reproductible.

### Semaine 6 (buffer)
- Réservée aux imprévus (téléchargements lents, debugging, retours du prof).

---

## 6. Structure de projet R

```
projet_sdm_engoulevent/
├── renv.lock
├── README.md
├── data/
│   ├── raw/           # téléchargements, lecture seule
│   └── processed/     # résultats intermédiaires
├── R/
│   ├── 01_select_mrc.R
│   ├── 02_download_ebird.R
│   ├── 03_filter_ebird.R
│   ├── 04_download_predictors.R
│   ├── 05_prepare_raster_stack.R
│   ├── 06_extract_covariates.R
│   ├── 07_vif_correlations.R
│   ├── 08_fit_maxent.R
│   ├── 09_fit_rf.R
│   ├── 10_block_cv.R
│   ├── 11_predict_maps.R
│   └── 12_figures.R
├── output/
│   ├── figures/
│   ├── maps/
│   └── tables/
└── rapport/
    └── rapport.qmd    # Quarto, exporte en PDF
```

---

## 7. Sources de données — résumé opérationnel

| Donnée | Source | URL | Taille approx. (1 MRC) |
|--------|--------|-----|------------------------|
| eBird EBD + sampling | Cornell Lab (demande d'accès) | https://ebird.org/data/download | 200–500 MB filtré QC |
| Carte écoforestière | Données Québec | https://www.donneesquebec.ca/recherche/dataset/carte-ecoforestiere-avec-perturbations | 50–200 MB |
| LiDAR dérivés (MHC, TWI) | Données Québec | https://www.donneesquebec.ca/recherche/dataset/produits-derives-de-base-du-lidar | 20–80 tuiles × ~30 MB |
| CHELSA Bio10, Bio18 | CHELSA | https://chelsa-climate.org/ | <20 MB (clip MRC) |
| Réseau routier | Adresses Québec | Données Québec | <50 MB |
| Atlas oiseaux nicheurs (validation) | AONQ | https://atlas-oiseaux.qc.ca | accès sur inscription |

**Budget total estimé : ~3–5 GB de données brutes, ~500 MB après preprocessing à 250 m.** Faisable sur ordinateur portable.

---

## 8. Livrables finaux

1. **Rapport** (Quarto/PDF, ~15–20 pages) : intro, méthodes, résultats, discussion.
2. **Dépôt GitHub** avec `renv` pour reproductibilité.
3. **Cartes** (4) : probabilité MaxEnt, probabilité RF, désaccord entre modèles, MESS.
4. **Tableau** des 5 zones favorables prédites non échantillonnées (avec coordonnées).
5. **Présentation orale** (10–15 min) si requise.

---

## 9. Risques et plans de mitigation

| Risque | Probabilité | Mitigation |
|--------|-------------|------------|
| Délai accès EBD | Moyenne | Demander dès semaine 1, démarrer en // sur les variables environnementales |
| LiDAR couverture incomplète sur la MRC | Moyenne | Vérifier en S1 ; à défaut, fallback sur hauteur dominante de la carte écoforestière |
| Trop peu de présences en juin-juillet | Faible (2500+ dispo) | Élargir à mai-août si nécessaire |
| Performance RF >> MaxEnt sans contraste écologique | Possible | Garder la comparaison comme contribution méthodologique, pas comme verdict |
| Surajustement | Probable si pas attentif | VIF strict, blockCV obligatoire, ratio ≥10 obs/variable respecté |

---

## 10. Ce qu'on **ne fait pas** (et pourquoi)

- **Pas d'occupancy model** (`unmarked`) : trop complexe pour POC 4–5 sem. À mentionner en discussion comme voie d'approfondissement (Johnston et al. 2021).
- **Pas d'ensemble model** (`biomod2`) : intéressant mais consomme du temps qu'on met mieux sur la validation et la discussion.
- **Pas de projection climat futur** : hors cadre POC.
- **Pas de variables de Niveau 3** (GPP MODIS, configuration avancée, ERA5) : excès de complexité pour le gain attendu.

---

## 11. Références clés à citer

- Johnston et al. (2021). *Diversity and Distributions*, 27, 1265–1277. — Référence méthodologique eBird.
- Strimas-Mackey et al. (2023). *Best Practices for Using eBird Data*. — Tutoriel R officiel.
- Guisan, Thuiller & Zimmermann (2017). *Habitat Suitability and Distribution Models*. — Manuel de référence.
- Valavi et al. (2019). blockCV. *Methods in Ecology and Evolution*, 10, 225–232.
- Wilson & Watts (2008). *Wilson Journal of Ornithology*, 120, 778–783. — Engoulevent, échelles paysagères.
- ECCC (2018). Programme de rétablissement Engoulevent. — Contexte conservation.

---

*Document de cadrage v1 — à raffiner après la sélection finale de la MRC en semaine 1.*
