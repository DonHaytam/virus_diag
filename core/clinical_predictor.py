"""core/clinical_predictor.py — Couche de diagnostic (prédiction).

ClinicalPredictor entoure un modèle DÉJÀ ENTRAÎNÉ et n'effectue AUCUN
apprentissage : entraîner le modèle est le rôle de pipeline/trainer.py (Trainer).

Concept POO illustré ici : le POLYMORPHISME (duck typing).
ClinicalPredictor n'impose aucun type de modèle précis ; il exige seulement
que l'objet possède une méthode predict(patient_data). N'importe quel modèle
(régression logistique, réseau de neurones, modèle sklearn, ...) fonctionne
sans modifier cette classe.
"""

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)


class ClinicalPredictor:
    """Sert un diagnostic clinique à partir d'un modèle déjà entraîné.

    Le modèle fourni peut être n'importe quel objet à condition d'exposer
    une méthode predict(patient_data) renvoyant une probabilité d'infection.
    """

    SEUIL_DECISION = 0.5  # au-delà de ce seuil, le patient est considéré infecté

    def __init__(self, model, seuil=SEUIL_DECISION):
        if not callable(getattr(model, "predict", None)):
            raise TypeError("Le modèle fourni doit implémenter une méthode predict(patient_data).")
        self.model = model      # modèle reçu DÉJÀ entraîné : on ne l'entraîne pas ici
        self.seuil = seuil

    def predict_proba(self, patient_data):
        """Retourne la probabilité d'infection estimée pour un patient."""
        return float(self.model.predict(patient_data))

    def diagnose(self, patient_data):
        """Retourne "Infecté" si la prédiction >= 0.5, sinon "Sain"."""
        if self.predict_proba(patient_data) >= self.seuil:
            return "Infecté"
        return "Sain"

    def diagnostiquer_lot(self, patients):
        """Diagnostique une liste de patients d'un seul appel."""
        return [self.diagnose(patient) for patient in patients]


# ---------------------------------------------------------------------------
# Démonstration : DEUX modèles différents fonctionnent avec la MÊME classe.
# (illustre le polymorphisme / "duck typing" : s'ils ont predict(), ça suffit)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    class FakeLogisticRegression:
        """Faussaire : imite une régression logistique déjà entraînée."""

        def predict(self, patient_data):
            # patient_data = [age, temperature, frequence_cardiaque, test_serologique]
            return 0.9 if patient_data[1] > 38.5 else 0.2

    class FakeNeuralNetwork:
        """Faussaire : imite un réseau de neurones déjà entraîné."""

        def predict(self, patient_data):
            return float(patient_data[3])  # proba = résultat direct du test sérologique

    patient_a = [45, 39.2, 105, 1]  # fièvre élevée, test positif
    patient_b = [22, 36.8, 72, 0]   # pas de fièvre, test négatif

    for modele in (FakeLogisticRegression(), FakeNeuralNetwork()):
        predictor = ClinicalPredictor(modele)
        print(f"--- Modèle : {type(modele).__name__} ---")
        print(f"Patient A -> {predictor.diagnose(patient_a)}")
        print(f"Patient B -> {predictor.diagnose(patient_b)}")
