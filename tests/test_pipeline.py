"""Tests du pipeline d'entraînement et d'évaluation."""

import pytest

from core.dataset import Dataset
from core.logistic_regression import RegressionLogistique
from pipeline.evaluator import Evaluator, comparer_modeles
from pipeline.trainer import Trainer, construire_modele, entrainer_depuis_csv


class ModeleFixe:
    """Modèle factice pour tester l'évaluateur sans entraînement."""

    nom = "Modèle fixe"

    def predict(self, X):
        return [0.9 if ligne[0] >= 0.5 else 0.1 for ligne in X]


def test_construire_modele():
    assert isinstance(construire_modele("logistique"), RegressionLogistique)
    assert construire_modele("logistic").nom == "Régression logistique"
    assert construire_modele("neurones").nom == "Réseau de neurones"
    assert construire_modele("MLP").nom == "Réseau de neurones"
    with pytest.raises(ValueError, match="inconnu"):
        construire_modele("foret")


def test_trainer_entraine():
    dataset = Dataset([[0.0], [0.1], [0.9], [1.0]], [0, 0, 1, 1], nom="test")
    modele = RegressionLogistique(lr=1.0)
    trainer = Trainer(modele, epochs=200, verbose=False)
    resultat = trainer.entrainer(dataset)
    assert resultat is modele
    assert modele.predict([0.0]) < 0.2
    assert modele.predict([1.0]) > 0.8


def test_evaluator_metriques():
    dataset = Dataset([[0.0], [0.2], [0.8], [1.0]], [0, 0, 1, 1], nom="test")
    resultats = Evaluator(ModeleFixe()).evaluer(dataset, verbose=False)
    assert resultats["modele"] == "Modèle fixe"
    assert resultats["predictions"] == [0, 0, 1, 1]
    assert resultats["exactitude"] == 1.0
    assert resultats["f1"] == 1.0
    assert resultats["matrice"] == {"vp": 2, "vn": 2, "fp": 0, "fn": 0}


def test_comparer_modeles():
    resultats = {
        "A": {"exactitude": 0.9, "precision": 0.8, "rappel": 0.7, "f1": 0.75},
        "B": {"exactitude": 0.8, "precision": 0.7, "rappel": 0.6, "f1": 0.65},
    }
    tableau = comparer_modeles(resultats, titre="Comparaison")
    assert "Comparaison" in tableau
    assert "A" in tableau and "B" in tableau
    assert "0.9000" in tableau


def test_entrainer_depuis_csv(chemin_csv):
    modele, dataset, train, test = entrainer_depuis_csv(
        chemin_csv, "logistique", verbose=False
    )
    assert len(dataset) == 693
    assert len(train) + len(test) == len(dataset)
    assert modele.poids is not None

    resultats = Evaluator(modele).evaluer(test, verbose=False)
    assert 0.0 <= resultats["exactitude"] <= 1.0
    assert resultats["exactitude"] >= 0.80
