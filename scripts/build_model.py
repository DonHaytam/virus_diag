"""scripts/build_model.py — Construit l'artefact du modèle de référence.

Le projet ne possède ni Dockerfile ni configuration de packaging. Le build
adapté consiste donc à entraîner le modèle de référence sur le jeu de données
du projet, à contrôler sa qualité, puis à sérialiser ses paramètres et ses
métriques dans un artefact JSON réutilisable par l'étape de déploiement.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from pipeline.evaluator import Evaluator
from pipeline.trainer import entrainer_depuis_csv

SEUIL_EXACTITUDE = 0.80


def construire_artefact(chemin_csv, chemin_sortie, seuil_exactitude=SEUIL_EXACTITUDE):
    """Entraîne le modèle, vérifie son exactitude et écrit l'artefact JSON."""
    modele, dataset, _, test = entrainer_depuis_csv(
        chemin_csv, "logistique", verbose=False
    )
    resultats = Evaluator(modele).evaluer(test, verbose=False)

    exactitude = resultats["exactitude"]
    if exactitude < seuil_exactitude:
        raise RuntimeError(
            f"Exactitude insuffisante : {exactitude:.4f} < {seuil_exactitude:.4f}"
        )

    artefact = {
        "projet": "virus_diag",
        "version_format": 1,
        "modele": modele.nom,
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "date_construction": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": dataset.noms_features,
        "normalisation": [
            {"min": minimum, "max": maximum}
            for minimum, maximum in dataset.stats
        ],
        "seuil_decision": 0.5,
        "parametres": {
            "poids": [float(poids) for poids in modele.poids],
            "biais": float(modele.biais),
        },
        "metriques": {
            "exactitude": exactitude,
            "precision": resultats["precision"],
            "rappel": resultats["rappel"],
            "f1": resultats["f1"],
        },
        "taille_jeu_test": len(test),
    }

    dossier = os.path.dirname(os.path.abspath(chemin_sortie))
    os.makedirs(dossier, exist_ok=True)
    with open(chemin_sortie, "w", encoding="utf-8") as fichier:
        json.dump(artefact, fichier, ensure_ascii=False, indent=2)
        fichier.write("\n")

    return artefact


def main(argv=None):
    parseur = argparse.ArgumentParser(description="Construit l'artefact du modèle virus_diag.")
    parseur.add_argument(
        "--csv",
        default=os.path.join(RACINE, "data", "patient_data.csv"),
        help="Chemin du jeu de données CSV.",
    )
    parseur.add_argument(
        "--sortie",
        default=os.path.join(RACINE, "build", "model_logistique.json"),
        help="Chemin de l'artefact JSON à produire.",
    )
    arguments = parseur.parse_args(argv)

    try:
        artefact = construire_artefact(arguments.csv, arguments.sortie)
    except RuntimeError as erreur:
        print(f"[build] ÉCHEC : {erreur}", file=sys.stderr)
        return 1

    metriques = artefact["metriques"]
    print("[build] Artefact généré :", arguments.sortie)
    print(f"[build] Modèle : {artefact['modele']}")
    print(
        "[build] Exactitude = {exactitude:.4f} | Précision = {precision:.4f} | "
        "Rappel = {rappel:.4f} | F1 = {f1:.4f}".format(**metriques)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
