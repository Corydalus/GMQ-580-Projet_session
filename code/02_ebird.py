"""02 — eBird : EBD filtré → zero-fill → variables de détection → extraction covariables.

Filtre les checklists (Johnston et al. 2021 : Stationary/Traveling, listes complètes,
durée <= 300 min, distance <= 5 km, <= 10 observateurs, juin–juillet), applique le
zero-fill, calcule les variables de détection (minutes_apres_coucher, phase_lune,
log_duree, jour_julien) et extrait les covariables du stack dans un buffer 30 m.
Lecture lazy via polars.scan_csv ; to_pandas() seulement à l'entrée sklearn.

Sortie : data/processed/table_modele.parquet
"""


def main() -> None:
    raise NotImplementedError("Branche feat/ebird — à implémenter.")


if __name__ == "__main__":
    main()
