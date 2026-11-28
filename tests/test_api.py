"""Tests de la couche applicative (API clinique)."""

import pytest

from app.api import APIClinique, construire_api
from core.clinical_predictor import ClinicalPredictor
from core.dataset import Dataset


class ModeleFixe:
    """Modèle factice renvoyant toujours la même probabilité."""

    nom = "Modèle fixe"

    def predict(self, X):
        return 0.85


@pytest.fixture
def api():
    dataset = Dataset(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        [0, 1],
        noms_features=["age", "temperature", "frequence_cardiaque", "test_serologique"],
        nom="test",
    ).normaliser()
    return APIClinique(ClinicalPredictor(ModeleFixe()), dataset)


def test_sante(api):
    etat = api.sante()
    assert etat["statut"] == "ok"
    assert etat["modele"] == "Modèle fixe"
    assert etat["seuil_decision"] == 0.5
    assert len(etat["features_attendues"]) == 4


def test_diagnostiquer(api):
    reponse = api.diagnostiquer({
        "age": 45,
        "temperature": 39.2,
        "frequence_cardiaque": 105,
        "test_serologique": 1,
    })
    assert reponse == {
        "diagnostic": "Infecté",
        "probabilite_infection": 0.85,
        "seuil_decision": 0.5,
    }


def test_champ_manquant(api):
    with pytest.raises(KeyError, match="Champs manquants"):
        api.diagnostiquer({"age": 45, "temperature": 39.2})


def test_valeur_non_numerique(api):
    with pytest.raises(ValueError):
        api.diagnostiquer({
            "age": "invalide",
            "temperature": 39.2,
            "frequence_cardiaque": 105,
            "test_serologique": 1,
        })


def test_construire_api_sur_donnees_reelles():
    api = construire_api("logistique", verbose=False)
    etat = api.sante()
    assert etat["statut"] == "ok"
    assert etat["modele"] == "Régression logistique"

    reponse = api.diagnostiquer({
        "age": 45,
        "temperature": 39.2,
        "frequence_cardiaque": 105,
        "test_serologique": 1,
    })
    assert reponse["diagnostic"] in ("Infecté", "Sain")
    assert 0.0 <= reponse["probabilite_infection"] <= 1.0
