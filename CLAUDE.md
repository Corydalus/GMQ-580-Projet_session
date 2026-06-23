# CLAUDE.md — GMQ-580 : SDM Engoulevent bois-pourri

> Instructions pour Claude Code. Ce fichier décrit le projet, les conventions et les tâches de setup à exécuter.

---

## 0. Contexte du projet

**Repo GitHub :** `https://github.com/Corydalus/GMQ-580-Projet_session`
**Espèce :** *Antrostomus vociferus* (Engoulevent bois-pourri)
**Objectif :** Modèle de distribution d'espèce (SDM) — Random Forest sur checklists eBird zero-fillées, prédiction à 5 m de résolution sur une MRC du Québec méridional.
**Stack :** Python · `uv` · `polars`, `geopandas`, `rasterio`, `rioxarray`, `scikit-learn`, `astral`, `matplotlib`, `contextily`, `tqdm`
**Protocole complet :** voir `Recherche documentaire/06_protocole_python_v2.md`

---

## 1. Tâche — Connexion et lecture du repo

```bash
# 1. Cloner le repo (si pas encore fait)
git clone https://github.com/Corydalus/GMQ-580-Projet_session.git
cd GMQ-580-Projet_session

# 2. Lire le README existant et en extraire :
#    - les instructions du professeur
#    - le modèle de README attendu
#    Puis confirmer la structure attendue avant de toucher quoi que ce soit.
cat README.md
```

---

## 2. Tâche — Structure des branches Git

Mettre en place la structure de branches suivante :

```
main          ← branche stable, livrables finaux uniquement
└── dev       ← intégration continue du travail en cours
    ├── feat/predictors   ← script 01_predictors.py
    ├── feat/ebird        ← script 02_ebird.py
    ├── feat/model        ← script 03_model.py
    ├── feat/predict      ← script 04_predict.py
    └── feat/figures      ← script 05_figures.py
```

```bash
git checkout -b dev
git push -u origin dev

for branch in predictors ebird model predict figures; do
  git checkout dev
  git checkout -b feat/$branch
  git push -u origin feat/$branch
done

git checkout dev
```

Règle : on ne merge dans `main` qu'un livrable hebdomadaire complet et fonctionnel.

---

## 3. Tâche — Arborescence du projet

Créer la structure de dossiers suivante (respecte `06_protocole_python_v2.md`) :

```
GMQ-580-Projet_session/
├── CLAUDE.md               ← ce fichier
├── README.md               ← README du prof (existant) + à compléter
├── pyproject.toml          ← dépendances uv
├── uv.lock                 ← généré par `uv sync`
├── .gitignore
│
├── data/
│   ├── raw/                ← téléchargements bruts (ignoré par Git)
│   ├── interim/            ← tuiles LiDAR agrégées à 5 m (ignoré par Git)
│   │   ├── MHC_5m/
│   │   ├── TWI_5m/
│   │   └── Pentes_5m/
│   └── processed/          ← stack_5m.tif, parquets (ignoré par Git sauf .parquet légers)
│
├── code/
│   ├── utils.py
│   ├── 01_predictors.py
│   ├── 02_ebird.py
│   ├── 03_model.py
│   ├── 04_predict.py
│   └── 05_figures.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── outputs/
│   ├── figures/            ← PNG exports
│   ├── maps/               ← GeoTIFF COG (ignoré par Git)
│   ├── tables/             ← CSV/parquet résultats
│   └── models/             ← rf.joblib (ignoré par Git)
│
└── rapport/
    └── rapport.md
```

```bash
mkdir -p data/raw data/interim/{MHC_5m,TWI_5m,Pentes_5m} data/processed
mkdir -p code notebooks outputs/{figures,maps,tables,models} rapport
touch code/utils.py code/01_predictors.py code/02_ebird.py \
      code/03_model.py code/04_predict.py code/05_figures.py
touch rapport/rapport.md
# Créer des .gitkeep pour les dossiers vides
touch data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep \
      outputs/maps/.gitkeep outputs/models/.gitkeep
```

---

## 4. Tâche — `.gitignore`

Créer `.gitignore` avec le contenu suivant :

```gitignore
# ── Données volumineuses ─────────────────────────────────────────────────────
data/raw/
data/interim/
data/processed/*.tif
data/processed/*.vrt

# Garder les petits parquets de résultats et les métadonnées
!data/processed/*.parquet
!data/processed/*.json

# ── Outputs lourds ───────────────────────────────────────────────────────────
outputs/maps/
outputs/models/
# Garder les figures légères et les tableaux CSV
!outputs/figures/*.png
!outputs/tables/

# ── Modèles sérialisés ───────────────────────────────────────────────────────
*.joblib
*.pkl

# ── Python ───────────────────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*.pyo
.Python
.venv/
.env
*.egg-info/
dist/
build/

# ── Jupyter ──────────────────────────────────────────────────────────────────
.ipynb_checkpoints/

# ── uv ───────────────────────────────────────────────────────────────────────
# uv.lock EST versionné (reproductibilité), .venv non
.venv/

# ── OS ───────────────────────────────────────────────────────────────────────
.DS_Store
Thumbs.db

# ── IDE ──────────────────────────────────────────────────────────────────────
.vscode/
.idea/
*.swp

# ── eBird — données sous accord de confidentialité ──────────────────────────
# Ne jamais versionner les fichiers EBD bruts
data/raw/ebird/
data/processed/ebird/*.txt
```

---

## 5. Tâche — `pyproject.toml`

Créer `pyproject.toml` :

```toml
[project]
name = "sdm-engoulevent"
version = "0.1.0"
description = "SDM Engoulevent bois-pourri — GMQ-580 projet de session"
requires-python = ">=3.11"

dependencies = [
    "polars>=0.20",
    "geopandas>=0.14",
    "rasterio>=1.3",
    "rioxarray>=0.15",
    "numpy>=1.26",
    "scipy>=1.12",
    "scikit-learn>=1.4",
    "astral>=3.2",
    "matplotlib>=3.8",
    "contextily>=1.5",
    "tqdm>=4.66",
    "joblib>=1.3",
    "pyarrow>=15",       # backend Polars parquet
    "pandas>=2.0",       # interface sklearn uniquement
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.0",
    "ipykernel>=6.0",
    "ruff>=0.3",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"
```

```bash
uv sync
```

---

## 6. Tâche — Compléter le README

Après avoir lu le README existant du professeur, compléter les sections manquantes en **conservant exactement** la structure imposée par le modèle du prof. Sections typiques à ajouter :

- Description du projet (espèce, zone, méthode)
- Données requises (eBird EBD, LiDAR Données Québec, CHELSA, carte écoforestière MFFP)
- Installation (`git clone` + `uv sync`)
- Utilisation (ordre d'exécution des scripts `01` → `05`)
- Structure du repo (arborescence)
- Références (Johnston 2021, Strimas-Mackey 2023, Guisan et al. 2017)

Ne pas effacer les instructions du professeur — les placer dans une section `## Instructions du cours`.

---

## 7. Commit initial

```bash
git add .gitignore pyproject.toml README.md CLAUDE.md
git add code/ notebooks/ rapport/
git add data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep
git add outputs/figures/.gitkeep outputs/tables/.gitkeep
git commit -m "feat: initialisation structure projet SDM Engoulevent"
git push origin dev
```

---

## 8. Conventions de code (à respecter dans tous les scripts)

- **CRS unique** : `EPSG:32198` (NAD83 / Québec Lambert) pour tout raster et vecteur.
- **Résolution** : 5 m (repli 10 m si RAM insuffisante — changer la constante `RESOLUTION_M` dans `utils.py`).
- **Lecture raster** : toujours fenêtrée (`rasterio.windows`) — jamais `.read()` sans window sur la mosaïque complète.
- **Lazy loading** : `polars.scan_csv()` pour l'EBD ; conversion `to_pandas()` uniquement à l'entrée sklearn.
- **COG** : tous les rasters de sortie en `driver="COG", compress="DEFLATE", blocksize=512`.
- **GDAL_CACHEMAX** : `os.environ["GDAL_CACHEMAX"] = "512"` dans chaque worker multiprocessing.
- **Reproductibilité** : `random_state=42` partout.
- **Style** : `ruff` (ligne 100 chars), docstring une ligne pour chaque fonction publique.
- **Buffer eBird** : extraction des covariables dans un buffer 30 m autour du point GPS (précision GPS eBird ≈ 5–30 m).

---

## 9. Variables du modèle (rappel)

| # | Variable | Source | Échelle |
|---|----------|--------|---------|
| 1 | Hauteur moy. canopée (MHC) | LiDAR Données Québec | 5 m → focal |
| 2 | TWI | LiDAR Données Québec | 5 m |
| 3 | Densité lisière forêt-ouvert | Carte écoforestière MFFP | focal 1 km |
| 4 | Proportion forêt feuillue/mélangée | Carte écoforestière MFFP | focal 500 m |
| 5 | Densité routes (km/km²) | Adresses Québec / RRN | focal 1 km |
| 6 | Température estivale (Bio10) | CHELSA v2.1 | 1 km → 5 m |

**Variables de détection (RF uniquement)** : `duree`, `minutes_apres_coucher`, `phase_lune`, `jour_julien`

---

*Ce fichier est maintenu par Alex Chêné — GMQ-580, Session 3.*
