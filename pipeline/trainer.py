"""pipeline/trainer.py — Entraînement du modèle.

Le Trainer a UNE responsabilité : entraîner un modèle à partir de données.
Il ne fait AUCUN diagnostic clinique : c'est le rôle de ClinicalPredictor
(core/clinical_predictor.py). Cette séparation est volontaire (voir Q3).
"""

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from core.dataset import Dataset
from core.logistic_regression import RegressionLogistique
from core.neural_network import ReseauNeurones


class Trainer:
    """Orchestre l'entraînement d'un modèle IA (n'importe lequel)."""

    def __init__(self, modele, lr=None, epochs=None, verbose=True):
        self.modele = modele
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

    def entrainer(self, dataset_train):
        """Entraîne le modèle sur un Dataset d'entraînement et le renvoie."""
        if self.verbose:
            print(f"[Trainer] Entraînement de « {self.modele.nom} » "
                  f"sur {len(dataset_train)} patients...")
        self.modele.entrainer(
            dataset_train.X,
            dataset_train.y,
            epochs=self.epochs,
            lr=self.lr,
            verbose=self.verbose,
        )
        if self.verbose:
            perte = self.modele.historique_perte[-1]
            print(f"[Trainer] Terminé. Perte finale = {perte:.4f}")
        return self.modele


def construire_modele(nom="logistique", **options):
    """Fabrique un modèle à partir d'un simple nom (utilisé par les app)."""
    nom = nom.lower()
    if nom in ("logistique", "logistic", "regression", "régression"):
        return RegressionLogistique(**options)
    if nom in ("neurones", "neural", "reseau", "réseau", "mlp"):
        return ReseauNeurones(**options)
    raise ValueError(f"Modèle inconnu : {nom}")


def entrainer_depuis_csv(chemin_csv, nom_modele="logistique", colonne_cible="infection",
                         ratio_test=0.2, seed=42, verbose=True):
    """Pipeline complet : CSV -> normalisation -> séparation -> entraînement.

    Renvoie (modele_entraine, dataset_complet, dataset_train, dataset_test).
    """
    dataset = Dataset.depuis_csv(chemin_csv, colonne_cible).normaliser()
    train, test = dataset.separer(ratio_test=ratio_test, seed=seed)
    modele = construire_modele(nom_modele)
    Trainer(modele, verbose=verbose).entrainer(train)
    return modele, dataset, train, test


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    chemin = os.path.join(RACINE, "data", "patient_data.csv")
    print("=== Entraînement de la régression logistique ===")
    modele_lr, dataset, train, test = entrainer_depuis_csv(chemin, "logistique")
    print()
    print("=== Entraînement du réseau de neurones ===")
    modele_nn, _, _, _ = entrainer_depuis_csv(chemin, "neurones")
    print()
    print(dataset.resume())
