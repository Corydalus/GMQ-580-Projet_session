#!/usr/bin/env python3
"""
GMQ405 - Projet de session - Téléchargement en boucle des données
===================================================================
Auteur : Alexandre Chéné, ********** VOS NOMS ************
Date   : 2026-06-10

OBJECTIF
--------
Télécharge en boucle les sources de données AUTOMATISABLES listées dans
03_plan_projet_session.md :

    1. LiDAR — produits dérivés MFFP (MHC, MNT, Pentes) par feuillet
       SNRC 1:20 000, à partir du CSV feuillets_zone_etude.csv.

Climat (température estivale) : PLUS de téléchargement en masse ici. La
température de surface estivale (LST) est désormais calculée À LA VOLÉE
par un pipeline STAC (Landsat Collection 2 niveau-2, Microsoft Planetary
Computer) dans code/01_predictors.py — voir README et CLAUDE.md. CHELSA
n'est plus utilisé.

Sources NON automatisées dans ce script (acquisition manuelle requise
ou URLs à trouver selon la région) :

    - eBird EBD + sampling : inscription Cornell Lab (déjà fait).
    - Atlas des oiseaux nicheurs : inscription QuébecOiseaux.
    - Carte écoforestière : voir le bloc TODO en bas du fichier.
    - Réseau routier Adresses Québec : voir TODO.
    - Milieux humides potentiels du Québec (MELCCFP, v2023) : voir TODO.

USAGE
-----
    # Par défaut : MNT + MHC + Pentes, 4 workers parallèles
    python telechargement_donnees.py \\
        --csv feuillets_zone_etude.csv \\
        --out data/raw

    # Aperçu sans télécharger
    python telechargement_donnees.py \\
        --csv feuillets_zone_etude.csv --out data/raw --dry-run

    # Sous-ensemble LiDAR
    python telechargement_donnees.py \\
        --csv feuillets_zone_etude.csv --out data/raw \\
        --products MHC,Pentes

DÉPENDANCES
-----------
    pip install requests tqdm

CONVENTIONS D'ARBORESCENCE
--------------------------
    Un seul niveau sous la racine, déterminé par le TYPE de fichier.
    Le code du feuillet reste dans le nom du fichier (pas de collision).

    data/raw/
    ├── MNT/               MNT_31I02SE.tif, MNT_31I01SO.tif, ...
    ├── MHC/               MHC_31I02SE.tif, MHC_31I01SO.tif, ...
    ├── Pentes/            Pentes_31I02SE.tif, ...
    └── milieux_humides/   milieux_humides_potentiels_2023.gpkg (manuel)

    (Climat : aucun fichier brut — LST calculée à la volée via STAC.)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

# Produits LiDAR : colonnes du CSV → choix par défaut.
# MNT_Ombre et Courbes_* sont surtout utiles pour la visualisation,
# pas pour le SDM. On les écarte par défaut pour économiser ~50 % du
# volume téléchargé.
LIDAR_PRODUITS_TOUS = ["MNT", "MHC", "MNT_Ombre",
                       "Pentes", "Courbes_GDB", "Courbes_GPKG"]
LIDAR_PRODUITS_DEFAUT = ["MNT", "MHC", "Pentes"]

# Climat : plus de CHELSA. La température de surface estivale (LST) est
# calculée à la volée via STAC (Landsat C2 L2) dans code/01_predictors.py.

# Réglages réseau
TIMEOUT_S        = 60
CHUNK_SIZE       = 1024 * 1024        # 1 MB
DEFAUT_WORKERS   = 4                   # téléchargements parallèles
N_RETRY          = 3
USER_AGENT       = "GMQ405-SDM-engoulevent/1.0 (academic)"


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("telechargement")


# ============================================================
# Type pour une tâche de téléchargement
# ============================================================

@dataclass
class Tache:
    url: str
    dest: Path
    source: str          # ex. 'lidar' / 'milieux_humides' — pour les stats
    feuillet: str = ""   # code SNRC 20K (LiDAR seulement) — pour les stats


# ============================================================
# Téléchargement unitaire (avec reprise + retry)
# ============================================================

def telecharger(tache: Tache, force: bool = False) -> str:
    """Télécharge un fichier. Retourne 'ok', 'skip' ou 'error'.

    - Si le fichier existe déjà ET a la bonne taille → 'skip'.
    - Écrit dans un .part puis renomme : pas de fichier corrompu si
      le script est interrompu.
    - Réessaie N_RETRY fois sur les erreurs réseau.
    """
    dest = tache.dest
    url = tache.url

    if dest.exists() and not force:
        # Skip si la taille locale correspond à Content-Length
        try:
            r = requests.head(url, timeout=TIMEOUT_S,
                              allow_redirects=True,
                              headers={"User-Agent": USER_AGENT})
            taille_attendue = int(r.headers.get("Content-Length", 0))
            if taille_attendue > 0 and dest.stat().st_size == taille_attendue:
                return "skip"
        except requests.RequestException:
            # Si HEAD échoue, on suppose que le fichier local est bon
            return "skip"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for tentative in range(1, N_RETRY + 1):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT_S,
                              headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(tmp, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True,
                    unit_divisor=1024, desc=dest.name,
                    leave=False, ncols=90,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            tmp.rename(dest)
            return "ok"
        except (requests.RequestException, IOError) as e:
            log.warning("Échec %d/%d pour %s : %s",
                        tentative, N_RETRY, dest.name, e)
            if tentative < N_RETRY:
                time.sleep(2 ** tentative)  # backoff exponentiel

    # Tous les essais ont échoué
    if tmp.exists():
        tmp.unlink()
    return "error"


# ============================================================
# Construction des listes de tâches
# ============================================================

def taches_lidar(csv_path: Path, out_dir: Path,
                 produits: list[str]) -> list[Tache]:
    """Lit le CSV des feuillets et produit une tâche par couple
    (feuillet × produit). Range : <out>/<produit>/<fichier>, c.-à-d. un
    seul niveau sous la racine, déterminé par le type de produit (MNT,
    MHC, Pentes, ...). Le code du feuillet reste dans le nom du fichier
    (ex. MNT_31I02SE.tif), donc pas de collision entre feuillets.
    """
    taches: list[Tache] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feuillet = row.get("Feuillet20K", "").strip()
            if not feuillet:
                continue
            for produit in produits:
                url = row.get(produit, "").strip()
                if not url:
                    continue
                nom = Path(urlparse(url).path).name
                dest = out_dir / produit / nom
                taches.append(Tache(url=url, dest=dest,
                                    source="lidar", feuillet=feuillet))
    return taches


# ============================================================
# TODO — Sources à compléter quand les URLs seront connues
# ============================================================
# Milieux humides potentiels du Québec (MELCCFP, v2023) :
#   - Page : https://www.donneesquebec.ca/recherche/fr/dataset/
#            milieux-humides-potentiels
#   - Distribution provinciale (GPKG / FGDB). Récupérer l'URL de la
#     ressource GPKG « Milieux humides potentiels 2023 » sur la page
#     Données Québec (le lien direct peut changer aux mises à jour), la
#     mettre dans taches_milieux_humides(), puis clip sur la MRC. CC-BY 4.0.
#
# Carte écoforestière à jour avec perturbations :
#   - Page : https://www.donneesquebec.ca/recherche/dataset/
#            carte-ecoforestiere-avec-perturbations
#   - Distribution par RÉGION ADMINISTRATIVE. Votre zone spanne 4
#     régions SNRC, qui correspondent grosso modo aux régions admin
#     suivantes au Québec :
#       * Trois-Rivières (31I) → Mauricie + Centre-du-Québec
#       * Québec        (21L) → Capitale-Nationale + Chaudière-Appalaches
#       * Montréal      (31H) → Montérégie + Laurentides + Lanaudière
#       * Sherbrooke    (21E) → Estrie
#   - URL par région : à récupérer manuellement sur la page Données
#     Québec (lien direct change avec les mises à jour annuelles), puis
#     ajouter une fonction `taches_ecoforestiere(out_dir, regions)`.
#
# Réseau routier Adresses Québec :
#   - Page : https://www.donneesquebec.ca/recherche/dataset/adresses-quebec
#   - Couverture provinciale → un seul gros téléchargement, puis clip.

def taches_ecoforestiere(out_dir: Path) -> list[Tache]:
    """À COMPLÉTER : remplir les URLs par région administrative."""
    urls_par_region: dict[str, str] = {
        # "mauricie":           "https://...",
        # "centre-du-quebec":   "https://...",
        # "capitale-nationale": "https://...",
        # "estrie":             "https://...",
        # "monteregie":         "https://...",
    }
    taches = []
    for region, url in urls_par_region.items():
        nom = Path(urlparse(url).path).name
        dest = out_dir / "ecoforestiere" / region / nom
        taches.append(Tache(url=url, dest=dest, source="ecoforestiere"))
    return taches


def taches_routes(out_dir: Path) -> list[Tache]:
    """À COMPLÉTER : URL Adresses Québec (couche routière)."""
    urls: list[str] = [
        # "https://www.donneesquebec.ca/.../adresses-quebec.zip",
    ]
    taches = []
    for url in urls:
        nom = Path(urlparse(url).path).name
        dest = out_dir / "routes" / nom
        taches.append(Tache(url=url, dest=dest, source="routes"))
    return taches


def taches_milieux_humides(out_dir: Path) -> list[Tache]:
    """À COMPLÉTER : URL de la ressource GPKG « Milieux humides potentiels
    2023 » (MELCCFP, Données Québec). Remplace l'ancienne source NHN/Canvec.
    """
    urls: list[str] = [
        # "https://www.donneesquebec.ca/.../milieux_humides_potentiels_2023.gpkg",
    ]
    taches = []
    for url in urls:
        nom = Path(urlparse(url).path).name
        dest = out_dir / "milieux_humides" / nom
        taches.append(Tache(url=url, dest=dest, source="milieux_humides"))
    return taches


# ============================================================
# Orchestration parallèle
# ============================================================

def lancer_en_parallele(taches: list[Tache], workers: int,
                        force: bool, dry_run: bool) -> dict[str, int]:
    """Exécute toutes les tâches en parallèle. Retourne un dict de
    stats par statut ('ok', 'skip', 'error')."""
    stats = {"ok": 0, "skip": 0, "error": 0}
    erreurs: list[str] = []

    if dry_run:
        log.info("DRY-RUN : %d fichiers seraient téléchargés. Aperçu :",
                 len(taches))
        for t in taches[:10]:
            log.info("  %-12s → %s", t.source, t.dest)
        if len(taches) > 10:
            log.info("  ... (+%d autres)", len(taches) - 10)
        return stats

    log.info("Démarrage : %d fichiers, %d workers parallèles.",
             len(taches), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(telecharger, t, force): t for t in taches}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Total", ncols=90):
            tache = futures[fut]
            try:
                status = fut.result()
            except Exception as e:
                status = "error"
                erreurs.append(f"{tache.dest.name} ({e})")
            stats[status] += 1
            if status == "error" and tache.url not in erreurs:
                erreurs.append(tache.url)

    log.info("Bilan : %d OK | %d skip (déjà présent) | %d erreurs",
             stats["ok"], stats["skip"], stats["error"])
    if erreurs:
        log.warning("Premières URLs en erreur :")
        for url in erreurs[:5]:
            log.warning("  - %s", url)
    return stats


# ============================================================
# CLI
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Téléchargement des données du projet SDM Engoulevent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="Chemin du CSV feuillets_zone_etude.csv")
    parser.add_argument("--out", type=Path, default=Path("data/raw"),
                        help="Dossier racine de sortie (défaut: data/raw)")
    parser.add_argument("--products", default=",".join(LIDAR_PRODUITS_DEFAUT),
                        help=("Produits LiDAR à télécharger, séparés par "
                              "virgule (défaut: %(default)s ; choix : "
                              + ",".join(LIDAR_PRODUITS_TOUS) + ")"))
    parser.add_argument("--skip-lidar", action="store_true",
                        help="Ne pas télécharger les fichiers LiDAR")
    parser.add_argument("--with-ecoforestiere", action="store_true",
                        help=("Inclure la carte écoforestière (suppose que "
                              "les URLs sont remplies dans le code)"))
    parser.add_argument("--with-routes", action="store_true",
                        help="Inclure le réseau routier (URLs à remplir)")
    parser.add_argument("--with-milieux-humides", action="store_true",
                        help=("Inclure les milieux humides potentiels 2023 "
                              "(URL à remplir dans le code)"))
    parser.add_argument("--workers", type=int, default=DEFAUT_WORKERS,
                        help=f"Workers parallèles (défaut: {DEFAUT_WORKERS})")
    parser.add_argument("--force", action="store_true",
                        help="Re-télécharger même si le fichier existe")
    parser.add_argument("--dry-run", action="store_true",
                        help="Lister sans télécharger")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    taches: list[Tache] = []

    # --- LiDAR ---
    if not args.skip_lidar:
        produits = [p.strip() for p in args.products.split(",") if p.strip()]
        invalides = set(produits) - set(LIDAR_PRODUITS_TOUS)
        if invalides:
            log.error("Produits invalides : %s. Choix : %s",
                      invalides, LIDAR_PRODUITS_TOUS)
            return 1
        if not args.csv.exists():
            log.error("CSV introuvable : %s", args.csv)
            return 1
        t_lidar = taches_lidar(args.csv, args.out, produits)
        n_feuillets = len({t.feuillet for t in t_lidar})
        log.info("LiDAR : %d feuillets × %d produits = %d fichiers",
                 n_feuillets, len(produits), len(t_lidar))
        taches.extend(t_lidar)

    # --- Climat (LST) : aucun téléchargement — calculé à la volée via STAC
    #     dans code/01_predictors.py (Landsat C2 L2, Planetary Computer).

    # --- Écoforestière (optionnel, URLs à compléter) ---
    if args.with_ecoforestiere:
        t_eco = taches_ecoforestiere(args.out)
        if not t_eco:
            log.warning("--with-ecoforestiere demandé mais aucune URL "
                        "n'est définie. Remplir taches_ecoforestiere().")
        else:
            log.info("Écoforestière : %d fichiers", len(t_eco))
            taches.extend(t_eco)

    # --- Routes (optionnel) ---
    if args.with_routes:
        t_rt = taches_routes(args.out)
        if not t_rt:
            log.warning("--with-routes demandé mais aucune URL définie.")
        else:
            log.info("Routes : %d fichiers", len(t_rt))
            taches.extend(t_rt)

    # --- Milieux humides potentiels (optionnel) ---
    if args.with_milieux_humides:
        t_mh = taches_milieux_humides(args.out)
        if not t_mh:
            log.warning("--with-milieux-humides demandé mais aucune URL "
                        "définie. Remplir taches_milieux_humides().")
        else:
            log.info("Milieux humides potentiels : %d fichiers", len(t_mh))
            taches.extend(t_mh)

    if not taches:
        log.warning("Aucune tâche. Vérifier les flags --skip-*.")
        return 0

    log.info("Destination racine : %s", args.out.resolve())
    stats = lancer_en_parallele(taches, args.workers,
                                args.force, args.dry_run)

    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
