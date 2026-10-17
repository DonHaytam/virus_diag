"""core/dataset.py — Chargement et représentation du jeu de données.

La classe Dataset encapsule les données (X, y), leurs noms de features et
les statistiques de normalisation. Elle sert de pont entre `utils`
(prétraitement) et les modèles du `core`.
"""

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from utils.preprocessing import (
    appliquer_normalisation,
    extraire_xy,
    normaliser,
    separer_train_test,
)


class Dataset:
    """Représente un jeu de données de patients (features X et cible y)."""

    def __init__(self, X, y, noms_features=None, nom="jeu_de_donnees", stats=None):
        self.X = X
        self.y = y
        self.noms_features = noms_features or [f"f{i}" for i in range(len(X[0]))]
        self.nom = nom
        self.stats = stats  # bornes (min, max) de normalisation, si calculées

    @classmethod
    def depuis_csv(cls, chemin, colonne_cible="infection", nom=None):
        """Construit un Dataset à partir d'un fichier CSV."""
        X, y, noms_features = extraire_xy(chemin, colonne_cible)
        return cls(X, y, noms_features, nom or os.path.basename(chemin))

    def normaliser(self):
        """Normalise les features dans [0, 1] et mémorise les bornes."""
        self.X, self.stats = normaliser(self.X, self.stats)
        return self

    def normaliser_patient(self, patient):
        """Normalise UN patient brut (liste de valeurs) avec les bornes apprises."""
        if self.stats is None:
            raise ValueError("Appelez d'abord Dataset.normaliser().")
        return appliquer_normalisation([patient], self.stats)[0]

    def separer(self, ratio_test=0.2, seed=42):
        """Renvoie deux Datasets : (entraînement, test)."""
        X_train, X_test, y_train, y_test = separer_train_test(self.X, self.y, ratio_test, seed)
        train = Dataset(X_train, y_train, self.noms_features, self.nom + " [train]", self.stats)
        test = Dataset(X_test, y_test, self.noms_features, self.nom + " [test]", self.stats)
        return train, test

    def prevalence(self):
        """Proportion de patients infectés dans le jeu de données."""
        return sum(self.y) / len(self.y) if self.y else 0.0

    def resume(self):
        return (f"Dataset '{self.nom}' : {len(self)} patients, "
                f"{len(self.noms_features)} features {self.noms_features}, "
                f"prévalence infection = {self.prevalence():.1%}")

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        return f"<Dataset {self.nom} n={len(self)}>"


if __name__ == "__main__":
    chemin = os.path.join(RACINE, "data", "patient_data.csv")
    dataset = Dataset.depuis_csv(chemin).normaliser()
    train, test = dataset.separer()
    print(dataset.resume())
    print(f"  -> entraînement : {len(train)} patients")
    print(f"  -> test         : {len(test)} patients")
