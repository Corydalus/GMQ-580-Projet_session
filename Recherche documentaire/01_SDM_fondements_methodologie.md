# SDM — Fondements méthodologiques et cadrage du projet

**Cours :** GMQ405 — Modélisation et analyse spatiale
**Espèce cible :** Engoulevent bois-pourri (*Antrostomus vociferus*) — alternative : Paruline azurée (*Setophaga cerulea*)
**Source de données :** eBird Basic Dataset (EBD) + sampling event data

---

## 1. Qu'est-ce qu'un SDM, formellement

Un **Species Distribution Model** corrèle des observations géoréférencées d'une espèce avec des variables environnementales pour estimer une **fonction de la niche réalisée**, puis projeter cette fonction dans l'espace géographique. Le cadre conceptuel canonique est celui de **Guisan, Thuiller & Zimmermann (2017)** qui décompose le travail en cinq étapes : (i) conceptualisation, (ii) préparation des données, (iii) ajustement du modèle, (iv) évaluation, (v) prédiction/projection.

Trois objectifs distincts — choisir lequel **avant** de commencer :

- **Inférence** : comprendre quelles variables structurent la distribution (interprétation des coefficients).
- **Interpolation/cartographie** : produire une carte de probabilité dans la zone d'étude.
- **Transfert/projection** : extrapoler dans l'espace (autre région) ou le temps (climat futur).

Ces trois objectifs n'imposent **pas** les mêmes contraintes méthodologiques. Votre projet est explicitement le 2e (cartographie au Québec) avec une composante d'inférence (quels habitats déterminent la distribution).

Distinction critique : **niche fondamentale** (tolérances physiologiques théoriques) vs **niche réalisée** (où l'espèce existe réellement, contrainte par les interactions biotiques et la dispersion). Un SDM corrélatif estime la niche réalisée — c'est une limite épistémique à reconnaître dans la discussion.

---

## 2. Les étapes du pipeline, en détail

### Étape 1 — Conceptualisation
Questions à fixer **avant** d'ouvrir RStudio :

- Quelle est la **fenêtre temporelle biologique** pertinente ? Pour l'Engoulevent bois-pourri : période de reproduction (juin–juillet), heures crépusculaires (≈ ±1h autour du coucher du soleil, possiblement pleine lune).
- Quelle est l'**extension spatiale** ? Aire de répartition connue au Québec (sud du Bouclier, Outaouais, Laurentides, Estrie) vs province entière.
- Quelle **résolution spatiale** ? Trade-off entre résolution des rasters (1 km, 250 m, 30 m) et précision GPS des checklists eBird (souvent ±100–500 m).
- L'unité d'analyse est-elle la **checklist** (point) ou un **carreau spatial agrégé** (ex. grille 3 km × 3 km comme dans Johnston et al. 2021) ?

### Étape 2 — Préparation des données

**Côté espèce (eBird) :**
- Téléchargement EBD + sampling event data via le portail eBird (demande d'accès requise).
- Filtrage avec le package **auk** : pays/province (`auk_country("CA")`, `auk_state("CA-QC")`), espèce, dates, type de protocole (Stationary, Traveling), `auk_complete()` pour ne garder que les checklists complètes (condition obligatoire pour le zero-filling).
- **Zero-filling** (`auk_zerofill()`) : transforme les checklists complètes sans détection en absences valides. C'est la clé de voûte qui distingue eBird des bases présence-seule comme GBIF.
- Filtres d'effort recommandés par Johnston et al. (2021) : durée ≤ 5h, distance ≤ 5 km, ≤ 10 observateurs — pour homogénéiser l'effort.
- **Sous-échantillonnage spatial** (spatial thinning) pour réduire le biais d'effort (plus d'observateurs en zone urbaine).

**Côté environnement :**
- **Climat** : WorldClim v2.1 ou CHELSA v2.1 (≈1 km, 19 variables bioclimatiques). CHELSA est généralement préféré pour le relief, mais WorldClim suffit pour le sud du Québec.
- **Couverture du sol** : MODIS MCD12Q1 (500 m, annuel), ESA WorldCover (10 m, 2020/2021), Canada AAFC Annual Crop Inventory, ou Cartographie écoforestière du MFFP (très précise pour habitat forestier — pertinente pour l'Engoulevent).
- **Topographie** : SRTM ou ASTER DEM → altitude, pente, exposition, TRI.
- **Indice de végétation** : NDVI MODIS ou Sentinel-2 (saisonnier).
- **Pour l'Engoulevent spécifiquement** : densité de chemins forestiers, proportion d'habitat ouvert/forêt clairsemée, distance aux barrens — variables suggérées par Wilson & Watts (Ontario).

### Étape 3 — Sélection des variables
- **Multicolinéarité** : calculer le VIF et retirer itérativement les variables avec VIF > 5 (ou > 10 selon le seuil retenu). Alternative : matrice de corrélation de Pearson/Spearman, retirer une variable de chaque paire avec |r| > 0.7.
- Réduire au **minimum biologiquement justifiable**. Règle empirique : ≥ 10 présences par variable retenue (sinon surajustement quasi garanti).
- Documenter **pourquoi** chaque variable est conservée (hypothèse écologique, pas data-driven uniquement).

### Étape 4 — Ajustement du modèle
Quatre familles méthodologiques principales :

| Famille | Données requises | Forces | Faiblesses |
|---------|------------------|--------|------------|
| **GLM/GAM** | Présence-absence | Interprétable, parcimonieux | Suppose forme fonctionnelle |
| **Random Forest / BRT (gbm)** | Présence-absence | Capture interactions non-linéaires | Boîte noire relative, risque surajustement |
| **MaxEnt** | Présence + background | Standard pour présence-seule | Sensible au choix du background |
| **Occupancy (unmarked)** | Détections répétées + covariables d'effort | **Sépare détection vs présence** | Hypothèse de fermeture, structure de données contraignante |

Pour eBird, **les occupancy models sont la voie royale** (Johnston et al. 2021, MacKenzie et al. 2002) : ils utilisent les covariables d'effort du sampling event data (`DURATION_MINUTES`, `EFFORT_DISTANCE_KM`, `NUMBER_OBSERVERS`, heure de début) pour modéliser la **détectabilité** séparément de l'**occupation**. Critique pour une espèce nocturne comme l'Engoulevent où la non-détection ≠ absence.

Le `Pao` (Probability of Absence given Observed) du modèle de Royle-Nichols ou l'occupancy classique de MacKenzie sont les deux variantes pertinentes — pas un modèle "Huggin–Burnham" (ce terme dans votre HTML semble être une confusion avec les modèles de capture-recapture de Huggins).

### Étape 5 — Évaluation
- **Validation spatiale croisée** : impérative. Le k-fold aléatoire surestime massivement la performance en présence d'autocorrélation spatiale. Utiliser **blockCV** (Valavi et al. 2019) pour générer des blocs spatialement séparés, avec une taille de bloc supérieure au range du variogramme empirique de la réponse.
- **Métriques** :
  - **AUC-ROC** : seuil-indépendant, intuitif mais critiqué pour présence-only.
  - **TSS** (True Skill Statistic) : seuil-dépendant, > 0.5 = bon.
  - **Boyce index** continu (CBI) : recommandé pour présence-seule, ∈ [-1, 1], > 0 = mieux que hasard.
  - **Calibration** : courbe de calibration (proba prédite vs fréquence observée).
- **Cohérence écologique** : les courbes de réponse partielles ont-elles un sens biologique ?

### Étape 6 — Prédiction et conversion en carte
C'est ici qu'on passe du **modèle** à la **carte prédictive** :

1. Préparer une **pile de rasters** des prédicteurs sur toute la zone de prédiction, alignée (même CRS, même résolution, même extent) — utiliser `terra::project()`, `terra::resample()`.
2. `terra::predict(raster_stack, model)` produit une carte continue de probabilité (ou d'occupancy).
3. **Quantifier l'incertitude** : bootstrap (≥ 100 réplicats), produire carte de l'écart-type ou intervalle de confiance par pixel.
4. **Évaluer l'extrapolation** : carte **MESS** (Multivariate Environmental Similarity Surface) — les pixels avec MESS < 0 sont hors du domaine environnemental d'entraînement, prédictions non fiables. Alternative plus moderne : ExDet ou MOP.
5. (Optionnel) Seuillage en carte binaire présence/absence — utiliser un seuil maximisant TSS, pas le 0.5 par défaut.

---

## 3. Questions à se poser AVANT de coder

**Cadrage scientifique :**
1. Quelle est notre **question de recherche unique** ? (Une seule phrase, sans "et".)
2. Quel est notre **public** ? (Conservation appliquée — MFFP, ECCC ? Académique ?)
3. Quelle **précision décisionnelle** est requise ? (Carte d'habitats prioritaires à 1 km vs identification de parcelles à 30 m.)

**Données :**
4. Combien de **présences eBird** disponibles pour l'Engoulevent au Québec dans la fenêtre temporelle ciblée ? (Si < 100–200, envisager modèles parcimonieux ou agrandir la fenêtre.)
5. Le **biais d'effort** est-il modélisable avec les covariables disponibles ? (Réponse pour eBird : oui, grâce au sampling event data.)
6. Avons-nous une **source d'absences indépendantes** pour validation externe ? (Atlas des oiseaux nicheurs du Québec, SOS-POP, inventaires nocturnes du MFFP.)

**Choix méthodologiques :**
7. Approche **single-model** ou **ensemble** (biomod2) ?
8. **Occupancy model** (rigoureux, contraignant) vs **encounter rate model** type Johnston et al. (plus flexible, balanced random forests) ?
9. **Résolution finale** et **CRS** ? (Recommandation : projection NAD83 / Quebec Lambert ou Statistics Canada Lambert, résolution 1 km pour modèle climatique, 250 m si on intègre couvert forestier MFFP.)
10. Comment **partager le code** et garantir la **reproductibilité** ? (renv, GitHub, structure de projet R standard.)

**Éthique et reconnaissance :**
11. Les données eBird ont des **conditions d'utilisation** (citation obligatoire de la version EBD, déclaration des usages). Pour un projet académique : déclaration auprès de eBird + Cornell Lab.

---

## 4. Contraintes spécifiques au projet

### Liées à l'espèce (Engoulevent bois-pourri)
- **Espèce nocturne/crépusculaire** : forte hétérogénéité dans la probabilité de détection selon l'heure, la phase lunaire, la météo. Inclure **heure de la checklist** (centrée sur le coucher du soleil) et idéalement phase lunaire comme covariables de détection.
- **Détection acoustique presque exclusive** : la durée d'écoute (`DURATION_MINUTES`) est un proxy critique de l'effort.
- **Échelle spatiale du domaine vital** ≈ 5–50 ha — un raster à 1 km peut sous-représenter la mosaïque d'habitats requise (forêt clairsemée + clairière + edge).
- **Données rares** : effectif de présences potentiellement limité, problème de **rare species SDM** (Lomba et al. 2010, Breiner et al. 2015 — ensembles of small models, ESM).

### Liées aux données eBird
- **Biais d'effort spatial** : concentration des observateurs près des routes, parcs nationaux, zones urbaines. Visualiser la densité de checklists comme première étape diagnostique.
- **Biais d'effort temporel** : pics les fins de semaine, en mai-juin pour la migration, peu d'effort nocturne.
- **Hétérogénéité observateur** : compétence variable. Filtrer sur observateurs expérimentés possible mais réduit n.
- **Erreurs de localisation** : protocole "Traveling" → la coordonnée est le départ, pas le centroïde. Considérer un buffer.

### Liées au cadre académique
- **Reproductibilité** : code R commenté, structure de projet (`renv`, scripts numérotés `01_download.R`, `02_clean.R`, etc.), README.
- **Temps disponible** : un SDM rigoureux avec validation spatiale et bootstrap demande des heures de calcul — anticiper.
- **Travail d'équipe** : partage Git/GitHub, division claire (préparation données / modélisation / cartographie / rédaction).

---

## 5. Stack R recommandé

| Étape | Packages |
|-------|----------|
| Extraction eBird | `auk` |
| Manipulation tabulaire | `dplyr`, `tidyr`, `lubridate` |
| Spatial (vecteur) | `sf` |
| Spatial (raster) | `terra` (préféré à `raster` désormais) |
| Climat | `geodata` (téléchargement WorldClim/CHELSA) |
| Occupancy | `unmarked`, `ubms` (version bayésienne avec Stan) |
| MaxEnt | `dismo` + `rJava` + Maxent.jar, ou `maxnet` (sans Java), tuning via `ENMeval` |
| Random Forest / BRT | `ranger`, `gbm`, `dismo::gbm.step` |
| Ensemble | `biomod2`, `sdm` |
| Validation spatiale | `blockCV` |
| Évaluation | `PresenceAbsence`, `ecospat` (Boyce, MESS) |
| Cartographie | `tmap`, `ggplot2` + `ggspatial` |

---

## 6. Références principales (vérifiables)

**Cadre théorique général**
- Guisan, A., Thuiller, W. & Zimmermann, N.E. (2017). *Habitat Suitability and Distribution Models: With Applications in R*. Cambridge University Press. — Manuel de référence.
- Elith, J. & Leathwick, J.R. (2009). Species distribution models: ecological explanation and prediction across space and time. *Annual Review of Ecology, Evolution, and Systematics*, 40, 677–697.
- Franklin, J. (2010). *Mapping Species Distributions: Spatial Inference and Prediction*. Cambridge University Press.

**eBird / données citoyennes**
- Johnston, A., Hochachka, W.M., Strimas-Mackey, M.E., Ruiz-Gutierrez, V., Robinson, O.J., Miller, E.T., Auer, T., Kelling, S. & Fink, D. (2021). Analytical guidelines to increase the value of community science data: An example using eBird data to estimate species distributions. *Diversity and Distributions*, 27, 1265–1277. **DOI: 10.1111/ddi.13271** — Article clé, à lire en premier.
- Strimas-Mackey, M., Hochachka, W.M., Ruiz-Gutierrez, V., Robinson, O.J., Miller, E.T., Auer, T., Kelling, S., Fink, D. & Johnston, A. (2023). *Best Practices for Using eBird Data*. Version 2.0. Cornell Lab of Ornithology. https://ebird.github.io/ebird-best-practices/
- Kéry, M. & Royle, J.A. (2016). *Applied Hierarchical Modeling in Ecology*, Vol. 1. Academic Press. — Référence pour les occupancy models.

**Occupancy et détection**
- MacKenzie, D.I., Nichols, J.D., Lachman, G.B., Droege, S., Royle, J.A. & Langtimm, C.A. (2002). Estimating site occupancy rates when detection probabilities are less than one. *Ecology*, 83, 2248–2255.
- Fiske, I. & Chandler, R. (2011). unmarked: An R package for fitting hierarchical models of wildlife occurrence and abundance. *Journal of Statistical Software*, 43, 1–23.
- Considerations for fitting occupancy models to data from eBird and similar volunteer-collected data. *Ornithology* (2023), 140(4), ukad035.

**Validation spatiale et évaluation**
- Valavi, R., Elith, J., Lahoz-Monfort, J.J. & Guillera-Arroita, G. (2019). blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods in Ecology and Evolution*, 10, 225–232.
- Hijmans, R.J. (2012). Cross-validation of species distribution models: removing spatial sorting bias and calibration with a null model. *Ecology*, 93, 679–688.
- Hirzel, A.H., Le Lay, G., Helfer, V., Randin, C. & Guisan, A. (2006). Evaluating the ability of habitat suitability models to predict species presences. *Ecological Modelling*, 199, 142–152. — Boyce index.

**MaxEnt et tuning**
- Muscarella, R., Galante, P.J., Soley-Guardia, M., Boria, R.A., Kass, J.M., Uriarte, M. & Anderson, R.P. (2014). ENMeval: An R package for conducting spatially independent evaluations and estimating optimal model complexity for MAXENT ecological niche models. *Methods in Ecology and Evolution*, 5, 1198–1205.

**Engoulevent bois-pourri**
- Environnement et Changement climatique Canada (2018). *Programme de rétablissement de l'Engoulevent bois-pourri (Antrostomus vociferus) au Canada*. Série de Programmes de rétablissement de la Loi sur les espèces en péril. https://www.canada.ca/en/environment-climate-change/services/species-risk-public-registry/recovery-strategies/eastern-whip-poor-will-2018.html
- COSEWIC (2022). *Assessment and Status Report on the Eastern Whip-poor-will Antrostomus vociferus in Canada*.
- Wilson, M.D. & Watts, B.D. (2008). Landscape configuration effects on distribution and abundance of Whip-poor-wills. *Wilson Journal of Ornithology*, 120, 778–783.
- Cink, C.L., Pyle, P. & Patten, M.A. (2020). Eastern Whip-poor-will (*Antrostomus vociferus*). In *Birds of the World* (P.G. Rodewald, Editor). Cornell Lab of Ornithology.

**Paruline azurée (alternative)**
- Environnement et Changement climatique Canada (2021). *Programme de rétablissement de la Paruline azurée (Setophaga cerulea) au Canada*. https://www.canada.ca/en/environment-climate-change/services/species-risk-public-registry/recovery-strategies/cerulean-warbler-2021.html

---

## 7. Prochaines étapes proposées

1. **Choisir définitivement l'espèce** en regardant le nombre brut de checklists avec détection au Québec (mai–juillet 2015–2025) — décision data-driven en 30 minutes avec `auk`.
2. **Définir l'unité d'analyse et la résolution** — décision conjointe avec l'équipe.
3. **Rédiger un document de cadrage** (1–2 pages) : question, hypothèses, méthode envisagée. Sert de référence stable pour l'équipe et le prof.
4. **Demander l'accès EBD** sur eBird (délai possible quelques jours).
5. **Mettre en place le projet R** avec `renv` et une structure de scripts numérotée.
