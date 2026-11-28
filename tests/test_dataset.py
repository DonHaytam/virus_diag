"""Tests de la classe Dataset."""

import pytest

from core.dataset import Dataset


def test_dataset_depuis_csv(chemin_csv):
    dataset = Dataset.depuis_csv(chemin_csv)
    assert len(dataset) == 693
    assert dataset.noms_features == ["age", "temperature", "frequence_cardiaque",
                                     "test_serologique"]
    assert set(dataset.y) <= {0, 1}
    assert 0.0 < dataset.prevalence() < 1.0
    assert "patient_data.csv" in dataset.resume()


def test_normalisation_et_patient():
    dataset = Dataset([[0.0, 10.0], [10.0, 20.0]], [0, 1],
                      noms_features=["a", "b"], nom="test")
    assert dataset.stats is None
    with pytest.raises(ValueError, match="normaliser"):
        dataset.normaliser_patient([5.0, 15.0])

    dataset.normaliser()
    assert dataset.stats == [(0.0, 10.0), (10.0, 20.0)]
    assert dataset.normaliser_patient([5.0, 15.0]) == [0.5, 0.5]


def test_separation_conserve_les_metadonnees():
    dataset = Dataset([[float(i), float(i * 2)] for i in range(10)],
                      [i % 2 for i in range(10)],
                      noms_features=["a", "b"], nom="jeu").normaliser()
    train, test = dataset.separer(ratio_test=0.2, seed=42)
    assert len(train) == 8
    assert len(test) == 2
    assert train.noms_features == dataset.noms_features
    assert train.stats == dataset.stats
    assert "train" in train.nom and "test" in test.nom


def test_representation():
    dataset = Dataset([[1.0]], [1], nom="un_patient")
    assert repr(dataset) == "<Dataset un_patient n=1>"
