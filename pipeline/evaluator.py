"""pipeline/evaluator.py — Évaluation des performances du modèle.

L'Evaluator compare les probabilités du modèle au seuil 0.5, puis calcule
les métriques (exactitude, précision, rappel, F1) via utils/metrics.py.
"""

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from utils.metrics import (
    exactitude,
    f1,
    matrice_confusion,
    precision,
    rappel,
    rapport_classification,
)


class Evaluator:
    """Évalue un modèle entraîné sur un jeu de test."""

    def __init__(self, modele, seuil=0.5):
        self.modele = modele
        self.seuil = seuil

    def evaluer(self, dataset_test, verbose=True):
        """Renvoie un dictionnaire de résultats (probas, prédictions, métriques)."""
        probas = self.modele.predict(dataset_test.X)
        predictions = [1 if p >= self.seuil else 0 for p in probas]

        resultats = {
            "modele": self.modele.nom,
            "seuil": self.seuil,
            "probas": probas,
            "predictions": predictions,
            "matrice": matrice_confusion(dataset_test.y, predictions),
            "exactitude": exactitude(dataset_test.y, predictions),
            "precision": precision(dataset_test.y, predictions),
            "rappel": rappel(dataset_test.y, predictions),
            "f1": f1(dataset_test.y, predictions),
        }
        if verbose:
            print(rapport_classification(
                dataset_test.y, predictions,
                titre=f"Rapport — {self.modele.nom} (seuil={self.seuil})",
            ))
        return resultats


def comparer_modeles(resultats_par_modele, titre="Comparaison des modèles"):
    """Fabrique un tableau lisible comparant plusieurs modèles évalués."""
    lignes = [
        titre,
        "-" * len(titre),
        f"{'Modèle':<26}{'Exactitude':>12}{'Précision':>12}{'Rappel':>10}{'F1':>10}",
    ]
    for nom, resultats in resultats_par_modele.items():
        lignes.append(
            f"{nom:<26}{resultats['exactitude']:>12.4f}{resultats['precision']:>12.4f}"
            f"{resultats['rappel']:>10.4f}{resultats['f1']:>10.4f}"
        )
    return "\n".join(lignes)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from pipeline.trainer import entrainer_depuis_csv

    chemin = os.path.join(RACINE, "data", "patient_data.csv")
    resultats = {}
    for nom in ("logistique", "neurones"):
        modele, _, _, test = entrainer_depuis_csv(chemin, nom, verbose=False)
        resultats[modele.nom] = Evaluator(modele).evaluer(test, verbose=True)
        print()

    print(comparer_modeles(resultats))
