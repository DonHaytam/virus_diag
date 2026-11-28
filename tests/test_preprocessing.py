"""Tests des fonctions de prétraitement des données."""

import pytest

from utils.preprocessing import (
    appliquer_normalisation,
    charger_csv,
    extraire_xy,
    nettoyer_lignes,
    normaliser,
    separer_train_test,
    statistiques_minmax,
)


def test_nettoyer_lignes_ecarte_invalides_et_doublons():
    entetes = ["age", "temperature"]
    lignes = [
        ["20", "36.5"],
        ["20", "36.5"],   # doublon
        ["30"],           # ligne incomplète
        ["", "37.0"],     # valeur manquante
        ["40", "38.0"],
    ]
    assert nettoyer_lignes(entetes, lignes) == [["20", "36.5"], ["40", "38.0"]]


def test_charger_csv(petit_csv):
    entetes, lignes = charger_csv(petit_csv)
    assert entetes == ["age", "temperature", "frequence_cardiaque",
                       "test_serologique", "infection"]
    assert len(lignes) == 6


def test_extraire_xy(petit_csv):
    X, y, noms = extraire_xy(petit_csv)
    assert noms == ["age", "temperature", "frequence_cardiaque", "test_serologique"]
    assert X[0] == [20.0, 36.5, 70.0, 0.0]
    assert y == [0, 0, 1, 1, 0, 1]
    assert len(X) == len(y) == 6


def test_extraire_xy_colonne_absente(petit_csv):
    with pytest.raises(ValueError, match="absente"):
        extraire_xy(petit_csv, colonne_cible="inexistante")


def test_statistiques_et_normalisation_minmax():
    X = [[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]]
    stats = statistiques_minmax(X)
    assert stats == [(0.0, 10.0), (10.0, 30.0)]

    X_norm, stats_retournees = normaliser(X)
    assert stats_retournees == stats
    assert X_norm == [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    assert all(0.0 <= valeur <= 1.0 for ligne in X_norm for valeur in ligne)


def test_appliquer_normalisation_borne_identique():
    # Une colonne constante ne doit pas provoquer de division par zéro.
    assert appliquer_normalisation([[5.0]], [(5.0, 5.0)]) == [[0.0]]


def test_separer_train_test_reproductible():
    X = [[float(i)] for i in range(10)]
    y = [i % 2 for i in range(10)]

    X_train, X_test, y_train, y_test = separer_train_test(X, y, ratio_test=0.2, seed=42)
    assert len(X_train) == len(y_train) == 8
    assert len(X_test) == len(y_test) == 2
    assert sorted(X_train + X_test) == sorted(X)

    second = separer_train_test(X, y, ratio_test=0.2, seed=42)
    assert second == (X_train, X_test, y_train, y_test)
