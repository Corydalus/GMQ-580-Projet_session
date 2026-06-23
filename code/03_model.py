"""03 — Modèle : Random Forest + tuning + validation spatiale + importance + PDP.

Entraîne un RandomForestClassifier(class_weight="balanced", random_state=42),
tuning par RandomizedSearchCV (20 itérations), validation GroupKFold sur blocs
spatiaux de 10 km (AUC-ROC + TSS sur 5 folds), importance par permutation et
courbes de réponse partielle (PDP) des 10 variables d'habitat.

Sorties : outputs/models/rf.joblib, métriques CV (outputs/tables/)
"""


def main() -> None:
    raise NotImplementedError("Branche feat/model — à implémenter.")


if __name__ == "__main__":
    main()
