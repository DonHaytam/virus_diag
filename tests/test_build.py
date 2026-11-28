"""Tests du script de construction de l'artefact de build."""

import json

import pytest

from scripts.build_model import construire_artefact


def test_construire_artefact(petit_csv, tmp_path):
    sortie = tmp_path / "build" / "model.json"
    artefact = construire_artefact(petit_csv, sortie, seuil_exactitude=0.0)

    assert sortie.exists()
    donnees = json.loads(sortie.read_text(encoding="utf-8"))
    assert donnees["projet"] == "virus_diag"
    assert donnees["modele"] == "Régression logistique"
    assert len(donnees["features"]) == 4
    assert len(donnees["normalisation"]) == 4
    assert len(donnees["parametres"]["poids"]) == 4
    assert set(donnees["metriques"]) == {"exactitude", "precision", "rappel", "f1"}
    assert artefact["metriques"]["exactitude"] == donnees["metriques"]["exactitude"]


def test_construire_artefact_refuse_qualite_insuffisante(petit_csv, tmp_path):
    with pytest.raises(RuntimeError, match="Exactitude insuffisante"):
        construire_artefact(petit_csv, tmp_path / "model.json", seuil_exactitude=1.1)
