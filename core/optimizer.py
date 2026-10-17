"""core/optimizer.py — Optimisation des paramètres du modèle.

L'optimiseur ne connaît PAS le modèle : il reçoit des paramètres et des
gradients (nombres, listes ou matrices imbriquées) et renvoie les paramètres
mis à jour. Il implémente une descente de gradient, avec momentum optionnel.
"""

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)


class DescenteDeGradient:
    """Descente de gradient (avec momentum) réutilisable par tout modèle.

    parametre <- parametre + momentum * vitesse - lr * gradient
    """

    def __init__(self, lr=0.1, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self._vitesses = {}

    def mise_a_jour(self, parametres, gradients, prefixe="p"):
        """Met à jour récursivement parametres selon gradients (même forme)."""
        if isinstance(parametres, list):
            return [
                self.mise_a_jour(p, g, f"{prefixe}[{i}]")
                for i, (p, g) in enumerate(zip(parametres, gradients))
            ]
        vitesse = self.momentum * self._vitesses.get(prefixe, 0.0) - self.lr * gradients
        self._vitesses[prefixe] = vitesse
        return parametres + vitesse

    def reinitialiser(self):
        """Remet le momentum à zéro (à faire au début d'un entraînement)."""
        self._vitesses.clear()
