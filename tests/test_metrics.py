"""Tests des métriques de classification."""

import pytest

from utils.metrics import (
    exactitude,
    f1,
    matrice_confusion,
    precision,
    rappel,
    rapport_classification,
)


@pytest.fixture
def donnees():
    # VP=1, VN=1, FP=1, FN=1
    return [1, 1, 0, 0], [1, 0, 1, 0]


def test_matrice_confusion(donnees):
    y_vrai, y_pred = donnees
    assert matrice_confusion(y_vrai, y_pred) == {"vp": 1, "vn": 1, "fp": 1, "fn": 1}


def test_metriques(donnees):
    y_vrai, y_pred = donnees
    assert exactitude(y_vrai, y_pred) == 0.5
    assert precision(y_vrai, y_pred) == 0.5
    assert rappel(y_vrai, y_pred) == 0.5
    assert f1(y_vrai, y_pred) == 0.5


def test_metriques_cas_parfait():
    y_vrai = [1, 0, 1, 0]
    assert exactitude(y_vrai, y_vrai) == 1.0
    assert precision(y_vrai, y_vrai) == 1.0
    assert rappel(y_vrai, y_vrai) == 1.0
    assert f1(y_vrai, y_vrai) == 1.0


def test_metriques_divisions_par_zero():
    assert exactitude([], []) == 0.0
    assert precision([0, 0], [0, 0]) == 0.0
    assert rappel([1, 1], [0, 0]) == 0.0
    assert f1([0, 0], [0, 0]) == 0.0


def test_rapport_classification(donnees):
    y_vrai, y_pred = donnees
    rapport = rapport_classification(y_vrai, y_pred, titre="Test")
    assert "Test" in rapport
    assert "VP=1" in rapport
    assert "Exactitude : 0.5000" in rapport
    assert "F1-score   : 0.5000" in rapport
