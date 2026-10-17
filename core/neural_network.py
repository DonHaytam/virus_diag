"""core/neural_network.py — Implémentation d'un réseau de neurones.

Petit perceptron multicouche (1 couche cachée) entraîné par rétropropagation
du gradient. Comme il hérite de ModeleIA, son interface est identique à celle
de la régression logistique : c'est le polymorphisme qui permet au
ClinicalPredictor de fonctionner avec l'un OU l'autre sans modification.
"""

import math
import os
import random
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from core.model import ModeleIA
from core.optimizer import DescenteDeGradient


def sigmoide(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class ReseauNeurones(ModeleIA):
    """Perceptron multicouche : n_features -> 1 couche cachée -> 1 sortie."""

    nom = "Réseau de neurones"

    def __init__(self, neurones_cache=8, lr=0.5, epochs=300, batch_size=32, optimizer=None, seed=42):
        self.neurones_cache = neurones_cache
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer or DescenteDeGradient(lr=lr, momentum=0.9)
        self.seed = seed
        self.poids_entree = None   # matrice [neurones cachés][n_features]
        self.biais_cache = None    # [neurones cachés]
        self.poids_sortie = None   # [neurones cachés]
        self.biais_sortie = 0.0
        self.historique_perte = []

    def _initialiser(self, n_features):
        rng = random.Random(self.seed)
        self.poids_entree = [[rng.uniform(-0.5, 0.5) for _ in range(n_features)]
                             for _ in range(self.neurones_cache)]
        self.biais_cache = [0.0] * self.neurones_cache
        self.poids_sortie = [rng.uniform(-0.5, 0.5) for _ in range(self.neurones_cache)]
        self.biais_sortie = 0.0
        self.historique_perte = []

    def _avant(self, x):
        """Propagation avant : renvoie (activations couche cachée, probabilité)."""
        cache = [sigmoide(sum(w * xk for w, xk in zip(neurone, x)) + biais)
                 for neurone, biais in zip(self.poids_entree, self.biais_cache)]
        z_sortie = sum(w * a for w, a in zip(self.poids_sortie, cache)) + self.biais_sortie
        return cache, sigmoide(z_sortie)

    def predict(self, X):
        if ModeleIA.est_un_echantillon_unique(X):
            return self._avant(X)[1]
        return [self._avant(x)[1] for x in X]

    def entrainer(self, X, y, epochs=None, lr=None, verbose=False):
        epochs = epochs or self.epochs
        if lr is not None:
            self.optimizer.lr = lr

        n_features = len(X[0])
        if self.poids_entree is None or len(self.poids_entree[0]) != n_features:
            self._initialiser(n_features)
        self.optimizer.reinitialiser()

        n = len(X)
        for epoch in range(1, epochs + 1):
            perte_totale = 0.0
            for debut in range(0, n, self.batch_size):
                lot_X = X[debut:debut + self.batch_size]
                lot_y = y[debut:debut + self.batch_size]

                grad_w1 = [[0.0] * n_features for _ in range(self.neurones_cache)]
                grad_b1 = [0.0] * self.neurones_cache
                grad_w2 = [0.0] * self.neurones_cache
                grad_b2 = 0.0

                for xi, yi in zip(lot_X, lot_y):
                    cache, proba = self._avant(xi)
                    delta2 = proba - yi
                    for j in range(self.neurones_cache):
                        grad_w2[j] += delta2 * cache[j]
                        delta1 = delta2 * self.poids_sortie[j] * cache[j] * (1 - cache[j])
                        for k in range(n_features):
                            grad_w1[j][k] += delta1 * xi[k]
                        grad_b1[j] += delta1
                    grad_b2 += delta2
                    perte_totale += -(yi * math.log(proba + 1e-12) +
                                      (1 - yi) * math.log(1 - proba + 1e-12))

                m = len(lot_X)
                grad_w1 = [[g / m for g in ligne] for ligne in grad_w1]
                grad_b1 = [g / m for g in grad_b1]
                grad_w2 = [g / m for g in grad_w2]
                grad_b2 /= m

                self.poids_entree = self.optimizer.mise_a_jour(self.poids_entree, grad_w1, "w1")
                self.biais_cache = self.optimizer.mise_a_jour(self.biais_cache, grad_b1, "b1")
                self.poids_sortie = self.optimizer.mise_a_jour(self.poids_sortie, grad_w2, "w2")
                self.biais_sortie = self.optimizer.mise_a_jour(self.biais_sortie, grad_b2, "b2")

            moyenne = perte_totale / n
            self.historique_perte.append(moyenne)
            if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
                print(f"    epoch {epoch:>4}/{epochs} - perte = {moyenne:.4f}")

        return self
