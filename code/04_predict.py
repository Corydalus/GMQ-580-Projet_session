"""04 — Prédiction : carte proba (fenêtrée) + incertitude + MESS + hotspots.

Applique le modèle RF au stack tuile par tuile pour produire la carte de
probabilité, la carte d'incertitude (variance inter-arbres), la carte MESS
(zones d'extrapolation) et l'identification des hotspots favorables à faible
effort d'échantillonnage eBird. Rasters de sortie en COG DEFLATE blocksize 512.

Sorties : outputs/maps/*.tif (COG), hotspots.gpkg
"""


def main() -> None:
    raise NotImplementedError("Branche feat/predict — à implémenter.")


if __name__ == "__main__":
    main()
