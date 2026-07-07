"""Chargement et validation typée de `config.yaml` (voir CLAUDE.md §3.6).

Une analyse = un fichier de config. Tous les scripts du pipeline importent la
configuration via `load_config()` / `load_config_from_cli()`, ce qui évite les
nombres magiques en dur et rend chaque analyse reproductible et citable.

    from config import load_config
    cfg = load_config()                 # config.yaml à la racine
    cfg = load_config("autre.yaml")     # ou un autre fichier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

CONFIG_DEFAUT = "config.yaml"


class _Base(BaseModel):
    # `extra="forbid"` : une clé inconnue (faute de frappe) fait échouer la validation.
    model_config = ConfigDict(extra="forbid")


class EspeceCfg(_Base):
    nom_latin: str
    nom_commun: str
    code_ebird: str


class ZoneEtudeCfg(_Base):
    gpkg: str
    crs: str = "EPSG:32198"
    resolution_m: float = Field(5, gt=0)

    @field_validator("crs")
    @classmethod
    def _crs_epsg(cls, v: str) -> str:
        if not v.upper().startswith("EPSG:"):
            raise ValueError("crs doit être de la forme 'EPSG:xxxx'")
        return v


def _mois_valides(v: list[int]) -> list[int]:
    if not v or any(m < 1 or m > 12 for m in v):
        raise ValueError("les mois doivent être des entiers dans 1..12")
    return v


class EbirdCfg(_Base):
    mois_saison: list[int]
    duree_max_min: float = Field(gt=0)
    distance_max_km: float = Field(gt=0)
    observateurs_max: int = Field(gt=0)
    protocoles: list[str]
    listes_completes: bool = True
    buffer_m: float = Field(30, gt=0)

    @field_validator("mois_saison")
    @classmethod
    def _mois(cls, v: list[int]) -> list[int]:
        return _mois_valides(v)


class ClimatStacCfg(_Base):
    collection: str = "landsat-c2-l2"
    fournisseur: str = "planetary-computer"
    mois: list[int]
    annees: list[int] | None = None
    couverture_nuageuse_max: float = Field(60, ge=0, le=100)
    reducteur: str = "median"

    @field_validator("mois")
    @classmethod
    def _mois(cls, v: list[int]) -> list[int]:
        return _mois_valides(v)

    @field_validator("reducteur")
    @classmethod
    def _reducteur(cls, v: str) -> str:
        if v not in {"median", "mean"}:
            raise ValueError("reducteur doit être 'median' ou 'mean'")
        return v


class ModeleCfg(_Base):
    random_state: int = 42
    bloc_cv_km: float = Field(10, gt=0)
    n_folds: int = Field(5, ge=2)
    n_iter_recherche: int = Field(20, ge=1)


class CheminsCfg(_Base):
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    outputs: str = "outputs"


class CalculCfg(_Base):
    n_workers: int | None = Field(None)  # None = auto (tous les cœurs)
    threads_per_worker: int = Field(2, ge=1)
    memory_limit: str = "auto"
    dashboard: bool = True
    processes: bool = True  # True = workers en processus (isolation) ; False = threads

    @field_validator("n_workers")
    @classmethod
    def _n_workers(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("n_workers doit être null (auto) ou un entier ≥ 1")
        return v


class VariablesCfg(_Base):
    routes_par_type: bool = False
    milieux_humides_par_type: bool = False


class Config(_Base):
    espece: EspeceCfg
    zone_etude: ZoneEtudeCfg
    ebird: EbirdCfg
    climat_stac: ClimatStacCfg
    modele: ModeleCfg
    chemins: CheminsCfg
    calcul: CalculCfg = CalculCfg()
    variables: VariablesCfg = VariablesCfg()


def load_config(path: str | Path = CONFIG_DEFAUT) -> Config:
    """Lit et valide un `config.yaml`. Lève FileNotFoundError / ValidationError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)


def load_config_from_cli(argv: list[str] | None = None) -> Config:
    """Résout `--config <chemin>` (défaut config.yaml) et charge la config."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=CONFIG_DEFAUT, type=Path)
    args, _ = parser.parse_known_args(argv)
    return load_config(args.config)


def main() -> None:
    """Valide `config.yaml` et affiche un résumé (utile en vérification rapide)."""
    cfg = load_config_from_cli()
    print("config.yaml valide ✅")
    print(f"  espèce      : {cfg.espece.nom_commun} ({cfg.espece.nom_latin})")
    print(f"  zone        : {cfg.zone_etude.gpkg} · {cfg.zone_etude.crs} · {cfg.zone_etude.resolution_m} m")
    print(f"  saison eBird: mois {cfg.ebird.mois_saison}")
    print(f"  STAC        : {cfg.climat_stac.collection} · {cfg.climat_stac.reducteur} · nuages ≤ {cfg.climat_stac.couverture_nuageuse_max}%")
    print(f"  calcul      : n_workers={cfg.calcul.n_workers} · threads={cfg.calcul.threads_per_worker}")


if __name__ == "__main__":
    main()
