"""core/model.py — Classe de base (contrat commun) de tous les modèles IA.

Tous les modèles du projet héritent de `ModeleIA`. Grâce à ce contrat commun
+ au polymorphisme, le `Trainer` et le `ClinicalPredictor` peuvent travailler
avec n'importe quel modèle sans connaître sa classe exacte.
"""

import os
import sys
from abc import ABC, abstractmethod

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)


class ModeleIA(ABC):
    """Contrat commun : entraîner les paramètres, puis prédire une probabilité."""

    nom = "ModeleIA"

    @abstractmethod
    def entrainer(self, X, y, epochs=None, lr=None, verbose=False):
        """Apprend les paramètres du modèle à partir des données (X, y)."""

    @abstractmethod
    def predict(self, X):
        """Renvoie la probabilité d'infection.

        - si X est UN patient (liste de nombres) -> float
        - si X est un LOT de patients (liste de listes) -> liste de floats
        """

    @staticmethod
    def est_un_echantillon_unique(X):
        """True si X ressemble à UN patient (liste de nombres)."""
        return len(X) > 0 and not isinstance(X[0], (list, tuple))

    def decrire(self):
        """Description courte du modèle (utile pour les rapports)."""
        return self.nom

    def __repr__(self):
        return f"<{type(self).__name__}>"
