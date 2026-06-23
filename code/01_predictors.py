"""01 — Prédicteurs : LiDAR → tuiles 5 m → VRT → variables paysagères → stack_5m.tif.

Produit le stack raster à 10 bandes (5 m, EPSG:32198, COG DEFLATE) à partir des
feuillets LiDAR (MHC, MNT, Pentes), de la carte écoforestière, du réseau routier,
des milieux humides et de CHELSA Bio10. Lecture fenêtrée uniquement (jamais la
mosaïque complète en RAM).

Sortie : data/processed/stack_5m.tif
"""


def main() -> None:
    raise NotImplementedError("Branche feat/predictors — à implémenter.")


if __name__ == "__main__":
    main()
