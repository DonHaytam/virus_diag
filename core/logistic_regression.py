"""core/logistic_regression.py — Implémentation de la régression logistique.

Modèle linéaire + fonction sigmoïde : renvoie une probabilité d'infection.
Il hérite de ModeleIA et expose donc predict(), ce qui suffit à ClinicalPredictor.
"""

import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from core.model import ModeleIA
from core.optimizer import DescenteDeGradient


def sigmoide(z):
    """Fonction d'activation sigmoïde, bornée pour éviter les débordements."""
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class RegressionLogistique(ModeleIA):
    """Régression logistique entraînée par descente de gradient."""

    nom = "Régression logistique"

    def __init__(self, lr=0.1, epochs=400, regularisation=0.0, optimizer=None):
        self.lr = lr
        self.epochs = epochs
        self.regularisation = regularisation
        self.optimizer = optimizer or DescenteDeGradient(lr=lr)
        self.poids = None            # un poids par feature
        self.biais = 0.0
        self.historique_perte = []

    def _initialiser(self, n_features):
        self.poids = [0.0] * n_features
        self.biais = 0.0
        self.historique_perte = []

    def _score(self, x):
        """Combinaison linéaire w.x + b."""
        return sum(w * xk for w, xk in zip(self.poids, x)) + self.biais

    def predict(self, X):
        if ModeleIA.est_un_echantillon_unique(X):
            return sigmoide(self._score(X))
        return [sigmoide(self._score(x)) for x in X]

    def entrainer(self, X, y, epochs=None, lr=None, verbose=False):
        epochs = epochs or self.epochs
        if lr is not None:
            self.optimizer.lr = lr

        n_features = len(X[0])
        if self.poids is None or len(self.poids) != n_features:
            self._initialiser(n_features)
        self.optimizer.reinitialiser()

        n = len(X)
        for epoch in range(1, epochs + 1):
            grad_poids = [0.0] * n_features
            grad_biais = 0.0
            perte = 0.0

            for xi, yi in zip(X, y):
                proba = sigmoide(self._score(xi))
                erreur = proba - yi
                for j in range(n_features):
                    grad_poids[j] += erreur * xi[j]
                grad_biais += erreur
                perte += -(yi * math.log(proba + 1e-12) + (1 - yi) * math.log(1 - proba + 1e-12))

            grad_poids = [g / n + self.regularisation * self.poids[j] for j, g in enumerate(grad_poids)]
            grad_biais /= n

            self.poids = self.optimizer.mise_a_jour(self.poids, grad_poids, "poids")
            self.biais = self.optimizer.mise_a_jour(self.biais, grad_biais, "biais")

            moyenne = perte / n
            self.historique_perte.append(moyenne)
            if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
                print(f"    epoch {epoch:>4}/{epochs} - perte = {moyenne:.4f}")

        return self
