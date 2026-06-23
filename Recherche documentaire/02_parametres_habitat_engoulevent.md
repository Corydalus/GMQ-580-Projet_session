# Paramètres potentiels pour l'analyse d'habitat de l'Engoulevent bois-pourri

**Espèce :** *Antrostomus vociferus* (Engoulevent bois-pourri)
**Période ciblée :** nidification au Québec (juin–juillet)
**Objectif :** identifier les variables environnementales candidates pour un SDM, leur justification écologique, l'échelle pertinente et la source de données associée.

---

## 1. Cadre écologique — ce qu'on apprend de la littérature

Quatre constats structurent le choix des paramètres :

1. **L'habitat se définit par la structure, pas par la composition.** Les études convergent : c'est l'**ouverture du couvert** et l'agencement vertical de la végétation qui conditionnent la présence, plus que les essences présentes (US Forest Service FEIS ; Ontario General Habitat Description). Un peuplement de pins, de chênes ou d'érables peut être suitable s'il offre la même structure.

2. **Mosaïque > peuplement homogène.** L'espèce sélectionne des **interfaces forêt-clairière** (densité de lisière élevée) avec des proportions intermédiaires de forêt à l'échelle du domaine vital (~49 % de couvert forestier au pic de sélection ; New York ; ace-eco.org/vol21/iss1/art5). Forêts continues fermées et milieux complètement ouverts sont évités tous les deux.

3. **Multi-échelle obligatoire.** L'analyse pertinente se fait à **au moins trois échelles** : (i) site de nidification (~100–200 m, structure verticale + sol), (ii) territoire de chasse (~500 m – 1 km, mosaïque + lisière), (iii) contexte paysager (3–5 km, connectivité, pression anthropique). Wilson & Watts (2008) et plusieurs études récentes montrent que les variables changent de signe ou perdent leur pouvoir explicatif selon l'échelle.

4. **Limites par les proies.** Insectivore aérien dépendant principalement des lépidoptères nocturnes (papillons de nuit) et des coléoptères. Les variables proxy de productivité d'insectes (NDVI, hétérogénéité paysagère, faible pression de pesticides, faible pollution lumineuse) ont du sens même si rarement quantifiées directement.

---

## 2. Paramètres candidats — par catégorie écologique

### 2.1 Structure du couvert forestier

| Paramètre | Justification écologique | Échelle | Source recommandée |
|---|---|---|---|
| **Densité du couvert (%)** | Niche optimale 26–83 % de couverture ; couvert intermédiaire = mosaïque | 100 m, 500 m, 1 km | LiDAR — Modèle de hauteur de canopée (MHC), Données Québec |
| **Hauteur moyenne de canopée (m)** | Proxy de l'âge et de la structure verticale | 100 m, 500 m | LiDAR MHC (Données Québec, résolution 1 m) |
| **Écart-type de hauteur de canopée** | Mesure de l'hétérogénéité structurelle verticale | 100 m, 500 m | Dérivé du LiDAR |
| **Densité de strate intermédiaire (% du couvert entre 2–10 m)** | Sous-étage clair recherché par l'espèce | 100 m | LiDAR retours classifiés |
| **Indice d'ouverture (gap fraction)** | Trouée et clairières dans la matrice forestière | 100 m, 500 m | LiDAR (analyse de trouées) |

### 2.2 Composition forestière et stade de succession

| Paramètre | Justification écologique | Échelle | Source recommandée |
|---|---|---|---|
| **Proportion forêt feuillue / mélangée / résineuse** | L'espèce niche typiquement sous canopée feuillue (préférence relative) mais utilise aussi pinèdes claires | 500 m, 1 km | Carte écoforestière du Québec méridional (MRNF) |
| **Classe d'âge des peuplements** | Forêts jeunes à mi-âge (régénération, succession précoce) favorisées | 500 m, 1 km | Carte écoforestière (champ CL_AGE) |
| **Classe de densité (A–D du MRNF)** | Densité B–C (couvert 41–80 %) attendue comme optimale | 100 m, 500 m | Carte écoforestière (champ CL_DENS) |
| **Présence de perturbations récentes (coupe, feu, chablis)** | Crée des trouées et conditions de succession précoce | 1 km | Carte écoforestière avec perturbations (MRNF, mise à jour annuelle) |
| **Proportion de jeunes plantations résineuses** | Pinèdes claires régulièrement utilisées (Pinery Park, Ontario) | 500 m, 1 km | Carte écoforestière |

### 2.3 Configuration paysagère

| Paramètre | Justification écologique | Échelle | Source recommandée |
|---|---|---|---|
| **Densité de lisière forêt-ouvert (m/ha)** | Variable la plus systématiquement positive dans la littérature | 500 m, 1 km, 3 km | Dérivée du couvert (FRAGSTATS, package `landscapemetrics` en R) |
| **Indice de Shannon de l'occupation du sol** | Hétérogénéité paysagère | 1 km, 3 km | Occupation des terres + métriques paysagères |
| **Proportion d'habitat ouvert (champs, friches, barrens)** | Aire de chasse au crépuscule | 500 m, 1 km | Cartographie de l'occupation des terres du Québec |
| **Distance à la plus proche grande zone ouverte (m)** | Accessibilité du territoire de chasse | point | Calcul GIS |
| **Taille moyenne des îlots forestiers (ha)** | Patches intermédiaires préférés | 1 km, 3 km | landscapemetrics |
| **Proportion agricole** | Négatif : pesticides, raréfaction des insectes | 1 km, 5 km | Inventaire annuel des cultures (AAFC) |
| **Proportion zones humides** | Productivité d'insectes ↑ | 500 m, 1 km | Cartographie milieux humides MELCCFP / Canards Illimités |

### 2.4 Topographie et substrat

| Paramètre | Justification écologique | Échelle | Source recommandée |
|---|---|---|---|
| **Élévation (m)** | Limite altitudinale au Québec ; effet indirect par climat | point + 1 km | MNT LiDAR 1 m (Données Québec) ou SRTM 30 m |
| **Pente (°)** | Substrats secs préférés sur reliefs modérés | 100 m, 500 m | Dérivé du MNT LiDAR |
| **Exposition (aspect en sinus/cosinus)** | Versants sud-ouest plus chauds, asséchants → barrens | 100 m, 500 m | Dérivé du MNT |
| **Indice topographique d'humidité (TWI)** | Identifie sols bien drainés (Engoulevent ↔ sols secs) | 100 m, 500 m | TWI dérivé du LiDAR (Données Québec) |
| **Indice de position topographique (TPI)** | Crêtes / replats vs vallées | 500 m | Dérivé du MNT |
| **Texture/drainage du sol** | Niche nidification = sols sableux/loameux bien drainés | point | Pédologie IRDA ; cartographie des sols du Québec |
| **Proximité de barrens / affleurements rocheux** | Habitat de prédilection (Ontario, Massachusetts) | point | Dépôts de surface (Données Québec) ; visualisation aerial |

### 2.5 Climat (covariables descriptives)

| Paramètre | Justification écologique | Échelle | Source recommandée |
|---|---|---|---|
| **Température moyenne période chaude (Bio10)** | Indice thermique de la saison de reproduction | 1 km | CHELSA v2.1 (préféré au Québec) ou WorldClim v2.1 |
| **Précipitations estivales (Bio18)** | Trop d'eau = sols saturés défavorables | 1 km | CHELSA / WorldClim |
| **Nombre de degrés-jours > 5 °C** | Activité des proies (lépidoptères) | 1 km | Données climatiques ECCC, MELCCFP |
| **Date moyenne de dégel** | Phénologie d'arrivée et de ponte | 1 km | Climate Atlas of Canada |

### 2.6 Productivité primaire et phénologie

| Paramètre | Justification écologique | Échelle | Source recommandée |
|---|---|---|---|
| **NDVI estival moyen (juin–juillet)** | Proxy de biomasse végétale et d'abondance d'insectes | 100 m, 500 m | Sentinel-2 (10 m, mensuel) ou MODIS MOD13Q1 (250 m, 16 j) |
| **Amplitude saisonnière de NDVI** | Hétérogénéité phénologique | 1 km | Séries temporelles MODIS |
| **Productivité primaire brute (GPP) MODIS** | Charge énergétique disponible | 500 m, 1 km | MOD17A2 (NASA, 500 m, 8 j) |

### 2.7 Pression anthropique

| Paramètre | Justification écologique | Échelle | Source recommandée |
|---|---|---|---|
| **Densité de routes (km/km²)** | Sources d'insectes (effet vortex) mais aussi mortalité routière | 1 km | Réseau routier national (RNR/NRCan) ou Adresses Québec |
| **Distance à la route la plus proche (m)** | Effet edge et perturbation | point | Calcul GIS |
| **Densité de bâti** | Fragmentation, dérangement | 1 km, 3 km | Bâtiments OpenStreetMap ; Adresses Québec |
| **Intensité de lumière nocturne (radiance VIIRS)** | Pollution lumineuse négative pour proies lépidoptères et désynchronise activité crépusculaire | 500 m, 1 km | VIIRS Day/Night Band (Earth Observation Group, ~500 m, annuel) |
| **Distance à la zone urbanisée majeure** | Pression cumulée | point | Statistique Canada — limites urbaines |

---

## 3. Variables de détection (à ne pas confondre avec l'habitat)

Ces variables modélisent la **probabilité de détection** dans un modèle d'occupancy. Elles ne décrivent pas la niche mais conditionnent la qualité de l'observation. Elles proviennent **directement** du fichier sampling d'eBird et n'exigent pas de couches géospatiales externes.

| Variable | Source | Effet attendu |
|---|---|---|
| Durée de la checklist (`DURATION_MINUTES`) | eBird sampling | + (plus on écoute, plus on détecte) |
| Distance parcourue (`EFFORT_DISTANCE_KM`) | eBird sampling | + (jusqu'à plateau) |
| Nombre d'observateurs (`NUMBER_OBSERVERS`) | eBird sampling | + |
| Heure de début (`TIME_OBSERVATIONS_STARTED`) → minutes après coucher du soleil | eBird + calcul (`suncalc`) | maximum au crépuscule |
| Fraction lunaire éclairée + altitude lune | calcul (`suncalc`) | + en nuit lunaire |
| Jour julien | calcul | pic d'activité fin juin – début juillet |
| Couverture nuageuse / précipitations à l'heure de la checklist | ERA5-Land (ECMWF) ou stations ECCC | – par mauvais temps |

---

## 4. Inventaire des sources de données spatiales

### 4.1 Sources québécoises (à privilégier)

| Jeu de données | Couverture | Résolution | Format | URL |
|---|---|---|---|---|
| **Carte écoforestière à jour (avec perturbations)** | Québec méridional | Polygones, échelle 1:20 000 | SHP / GPKG | donneesquebec.ca/recherche/dataset/carte-ecoforestiere-avec-perturbations |
| **Carte écoforestière originale + résultats d'inventaire** | Québec | Polygones | SHP | donneesquebec.ca/recherche/dataset/resultats-d-inventaire-et-carte-ecoforestiere |
| **LiDAR — Modèles numériques (MNT, MHC, pente, courbes)** | Québec (couverture progressive) | 1 m | TIFF | donneesquebec.ca/recherche/dataset/produits-derives-de-base-du-lidar |
| **Indice topographique d'humidité (TWI) issu du LiDAR** | Québec | 1 m | TIFF | Données Québec — LiDAR dérivés |
| **Écotone riverain issu du LiDAR** | Québec | Vecteur | SHP | donneesquebec.ca/recherche/dataset/ecotones-riverains-issus-du-lidar |
| **Cartographie de l'occupation des terres du Québec** | Québec méridional | 10 m | TIFF | donneesquebec.ca/recherche/dataset/cartographie-de-l-occupation-des-terres-du-quebec |
| **Cartographie des milieux humides** | Régions sud du QC | Polygones | SHP | MELCCFP / Canards Illimités Canada |
| **Adresses Québec — réseau routier** | Québec | Vecteur | SHP / GPKG | Données Québec |
| **Dépôts de surface** | Québec | Polygones | SHP | Données Québec |
| **Atlas des oiseaux nicheurs du Québec (AONQ)** | Québec | Maillage 10×10 km | Téléchargement après inscription | atlas-oiseaux.qc.ca |

### 4.2 Sources canadiennes complémentaires

| Jeu de données | Couverture | Résolution | URL |
|---|---|---|---|
| **Land Cover of Canada 2020** | Canada | 30 m | open.canada.ca (NRCan) |
| **Inventaire annuel des cultures (AAFC)** | Canada agricole | 30 m, annuel | open.canada.ca |
| **Réseau routier national (RRN/NRN)** | Canada | Vecteur | open.canada.ca |
| **Climat — normales 1991–2020** | Canada | Stations | climate.weather.gc.ca |
| **Climate Atlas of Canada** | Canada | 10 km | climateatlas.ca |
| **Hansen Global Forest Change** | Mondial | 30 m, annuel | glad.earthengine.app |

### 4.3 Sources mondiales

| Jeu de données | Résolution | Pertinence |
|---|---|---|
| **CHELSA v2.1** (bioclim) | 1 km | climat (préféré à WorldClim en milieu accidenté) |
| **WorldClim v2.1** | 1 km | climat, alternative |
| **ESA WorldCover 2021** | 10 m | couvert global, vérification croisée |
| **MODIS MOD13Q1 / MOD17A2** | 250–500 m | NDVI, GPP, séries temporelles longues |
| **Sentinel-2** | 10 m | NDVI mensuel, haute résolution |
| **SRTM 1 arc-sec** | 30 m | MNT pour zones non couvertes par LiDAR québécois |
| **VIIRS Day/Night Band — VNL** | ~500 m, annuel | pollution lumineuse (Earth Observation Group, Colorado School of Mines) |

### 4.4 Données biologiques pour validation externe

| Source | Description |
|---|---|
| Atlas des oiseaux nicheurs du Québec 2010–2014 (et 2025+) | Présence/absence par carreau, données indépendantes d'eBird |
| Centre de données sur le patrimoine naturel du Québec (CDPNQ) | Mentions officielles d'espèces à statut |
| Réseau de relevés des oiseaux nocturnes (NightjarSurveys) | Données structurées spécifiques aux caprimulgidés |
| Programme SOS-POP (Québec Oiseaux) | Suivi d'espèces sensibles |
| iNaturalist Canada | Observations supplémentaires (validation, peu de checklists complètes) |

---

## 5. Tableau de synthèse — priorisation

Trois niveaux selon le rapport (pouvoir prédictif attendu) / (coût d'acquisition) :

**Niveau 1 — incontournables, à inclure dès la première itération**

- Densité de couvert et hauteur de canopée (LiDAR Québec)
- Composition forestière (carte écoforestière)
- Densité de lisière forêt-ouvert (calculée à 500 m et 1 km)
- TWI (LiDAR Québec)
- Bio10 et Bio18 (CHELSA)
- Densité de routes
- Variables de détection eBird (effort + heure + lune)

**Niveau 2 — à ajouter en deuxième itération si le modèle de niveau 1 sous-performe**

- Indice de Shannon paysager
- Proportion agricole
- Pollution lumineuse VIIRS
- NDVI Sentinel-2 estival
- Classe d'âge des peuplements
- Texture de sol (pédologie IRDA)

**Niveau 3 — variables exploratoires ou pour analyses spécifiques**

- GPP MODIS
- Distance aux affleurements rocheux / barrens
- Métriques de configuration avancées (patch cohesion, contagion)
- Couverture nuageuse historique ERA5
- Dépôts de surface (proxy géomorphologique)

---

## 6. Recommandations méthodologiques

- **Calculer chaque variable à plusieurs échelles** (au moins 250 m, 1 km, 3 km) puis sélectionner par AIC ou par expertise écologique la meilleure échelle par variable (Wilson & Watts 2008 ; ace-eco.org/vol21/iss1/art5).
- **Réaliser une matrice de corrélation et un VIF** (seuil < 5) avant la modélisation finale. Beaucoup de variables LiDAR sont fortement intercorrélées.
- **Garder un nombre raisonnable de variables** : règle empirique ≥ 10 présences par variable retenue ; au-delà, le surajustement est quasi garanti pour cette espèce rare.
- **Vérifier la disponibilité LiDAR sur l'aire d'étude** avant de s'engager : la couverture du LiDAR québécois est progressive (toutes les régions sud sont quasi couvertes en 2026, mais vérifier dataset par dataset).
- **Ne pas confondre niveau site (modèle d'occupation) et niveau checklist (modèle de détection)** : seules les variables environnementales décrites en section 2 sont des covariables d'occupation ; celles de la section 3 sont uniquement covariables de détection.

---

## 7. Références principales utilisées

**Écologie de l'espèce**
- Cink, C. L., Pyle, P. & Patten, M. A. 2020. Eastern Whip-poor-will (*Antrostomus vociferus*). *Birds of the World*, Cornell Lab of Ornithology.
- COSEWIC. 2022. *Assessment and Status Report on the Eastern Whip-poor-will Antrostomus vociferus in Canada*. https://www.canada.ca/en/environment-climate-change/services/species-risk-public-registry/cosewic-assessments-status-reports/eastern-whip-poor-will-2022.html
- ECCC. 2018. *Programme de rétablissement de l'Engoulevent bois-pourri (Antrostomus vociferus) au Canada*. https://www.canada.ca/en/environment-climate-change/services/species-risk-public-registry/recovery-strategies/eastern-whip-poor-will-2018.html
- US Forest Service. FEIS — *Antrostomus vociferus*. https://research.fs.usda.gov/feis/species-reviews/anvo

**Sélection d'habitat et échelles spatiales**
- Wilson, M. D. & Watts, B. D. 2008. Landscape configuration effects on distribution and abundance of Whip-poor-wills. *Wilson Journal of Ornithology*, 120, 778–783.
- *Eastern Whip-poor-will Breeding Resource Selection and Nest Survival in Northern New York* (SUNY thesis). https://soar.suny.edu/handle/20.500.12648/16273
- *High variation in Eastern Whip-poor-will home-range size and shape limits the effectiveness of one-size-fits-all habitat protection methods*. Avian Conservation and Ecology, vol 20, iss 1. https://ace-eco.org/vol20/iss1/art14/
- *Home range size, overlap, and habitat selection of diurnally roosting Eastern Whip-poor-wills during the breeding season*. Avian Conservation and Ecology, vol 21, iss 1. https://ace-eco.org/vol21/iss1/art5/
- *Diurnal and nocturnal habitat preference of Eastern Whip-poor-wills in the northern portion of their breeding range*. https://www.researchgate.net/publication/354771882
- *Landscape features affect occupancy probability of Antrostomus vociferus in West Virginia managed forests*. Ornithological Applications. https://academic.oup.com/condor/advance-article/doi/10.1093/ornithapp/duaf037/8221656

**Lune, proies et pollution lumineuse**
- English, P. A., Mills, A. M., Cadman, M. D., et al. 2017. *Lunar synchronization of daily activity patterns in a crepuscular avian insectivore*. PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC7391349/
- Wilson, A. C., et al. 2024. *Moonlight drives the energy balance and annual cycle of a nocturnal forager*. *Science Advances*. https://www.science.org/doi/10.1126/sciadv.aed8204
- Boyes, D. H., et al. 2021. Is light pollution driving moth population declines? *Insect Conservation and Diversity*, 14, 167–187.
- Owens, A. C. S., et al. 2020. Light pollution is a driver of insect declines. *Biological Conservation*, 241, 108259.

**Sources de données**
- Données Québec — LiDAR : https://www.donneesquebec.ca/recherche/dataset/produits-derives-de-base-du-lidar
- Données Québec — carte écoforestière à jour : https://www.donneesquebec.ca/recherche/dataset/carte-ecoforestiere-avec-perturbations
- Données Québec — occupation des terres : https://www.donneesquebec.ca/recherche/dataset/cartographie-de-l-occupation-des-terres-du-quebec
- MRNF — Norme de stratification écoforestière (5e inventaire) : https://mrnf.gouv.qc.ca/documents/forets/inventaire/carto_5E_methodes_donnees.pdf
- Earth Observation Group — VIIRS Nighttime Lights : https://eogdata.mines.edu/products/vnl/
- CHELSA v2.1 : https://chelsa-climate.org/
