"""Tests des modèles, de l'optimiseur et de la couche de diagnostic."""

import pytest

from core.clinical_predictor import ClinicalPredictor
from core.logistic_regression import RegressionLogistique, sigmoide
from core.model import ModeleIA
from core.neural_network import ReseauNeurones
from core.optimizer import DescenteDeGradient


class FauxModele:
    """Modèle minimal respectant le contrat attendu par ClinicalPredictor."""

    nom = "Faux modèle"

    def __init__(self, probabilite):
        self.probabilite = probabilite

    def predict(self, X):
        return self.probabilite


def test_sigmoide():
    assert sigmoide(0.0) == 0.5
    assert 0.0 < sigmoide(-2.0) < 0.5
    assert 0.5 < sigmoide(2.0) < 1.0
    assert sigmoide(-1000.0) == 0.0
    assert sigmoide(1000.0) == 1.0


def test_modele_ia_est_abstrait():
    with pytest.raises(TypeError):
        ModeleIA()


def test_est_un_echantillon_unique():
    assert ModeleIA.est_un_echantillon_unique([1, 2, 3]) is True
    assert ModeleIA.est_un_echantillon_unique([[1, 2], [3, 4]]) is False
    assert ModeleIA.est_un_echantillon_unique([]) is False


def test_regression_logistique_apprend():
    X = [[0.0], [0.1], [0.9], [1.0]]
    y = [0, 0, 1, 1]
    modele = RegressionLogistique(lr=1.0, epochs=300)
    modele.entrainer(X, y)

    probas = modele.predict(X)
    assert probas[0] < 0.2
    assert probas[1] < 0.2
    assert probas[2] > 0.8
    assert probas[3] > 0.8
    assert modele.historique_perte[-1] < modele.historique_perte[0]


def test_regression_logistique_predict_lot_coherent():
    modele = RegressionLogistique(lr=0.5, epochs=50)
    modele.entrainer([[0.0], [1.0]], [0, 1])
    lot = modele.predict([[0.2], [0.8]])
    assert lot[0] == modele.predict([0.2])
    assert lot[1] == modele.predict([0.8])


def test_reseau_neurones_predict():
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0, 1, 1, 0]
    modele = ReseauNeurones(neurones_cache=4, lr=0.5, epochs=100, seed=7)
    modele.entrainer(X, y)

    probas = modele.predict(X)
    assert len(probas) == len(X)
    assert all(0.0 <= p <= 1.0 for p in probas)
    assert modele.predict(X[0]) == probas[0]
    assert modele.historique_perte[-1] < modele.historique_perte[0]


def test_optimiseur_descente_de_gradient():
    optimiseur = DescenteDeGradient(lr=0.1)
    assert optimiseur.mise_a_jour([1.0], [1.0], "p") == [0.9]
    assert optimiseur.mise_a_jour([1.0], [0.0], "p") == [1.0]

    optimiseur.reinitialiser()
    assert optimiseur._vitesses == {}


def test_optimiseur_momentum_et_structures_imbriquees():
    optimiseur = DescenteDeGradient(lr=0.1, momentum=0.9)
    premiere = optimiseur.mise_a_jour([[1.0, 1.0]], [[1.0, 0.0]], "w")
    assert premiere == [[0.9, 1.0]]
    seconde = optimiseur.mise_a_jour(premiere, [[0.0, 0.0]], "w")
    assert seconde == [[0.81, 1.0]]


def test_clinical_predictor_seuil_et_lot():
    predictor = ClinicalPredictor(FauxModele(0.6))
    assert predictor.predict_proba([0, 0, 0, 0]) == 0.6
    assert predictor.diagnose([0, 0, 0, 0]) == "Infecté"
    assert predictor.diagnostiquer_lot([[0], [0]]) == ["Infecté", "Infecté"]

    strict = ClinicalPredictor(FauxModele(0.6), seuil=0.7)
    assert strict.diagnose([0, 0, 0, 0]) == "Sain"


def test_clinical_predictor_refuse_modele_sans_predict():
    with pytest.raises(TypeError):
        ClinicalPredictor(object())
