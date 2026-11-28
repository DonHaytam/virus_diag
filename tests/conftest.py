"""Configuration partagée de la suite de tests pytest."""

import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))


@pytest.fixture
def chemin_csv():
    """Chemin du jeu de données réel du projet."""
    return RACINE / "data" / "patient_data.csv"


@pytest.fixture
def petit_csv(tmp_path):
    """Petit jeu de données déterministe pour les tests unitaires."""
    chemin = tmp_path / "patients.csv"
    chemin.write_text(
        "age,temperature,frequence_cardiaque,test_serologique,infection\n"
        "20,36.5,70,0,0\n"
        "30,36.8,72,0,0\n"
        "40,38.5,90,1,1\n"
        "50,39.0,95,1,1\n"
        "60,36.7,68,0,0\n"
        "70,38.9,92,1,1\n",
        encoding="utf-8",
    )
    return chemin
