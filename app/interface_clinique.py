"""app/interface_clinique.py — Interface destinée aux médecins.

C'est la couche « application » visible par l'utilisateur final. Elle saisit
des données patient, appelle le ClinicalPredictor et affiche le diagnostic.
Aucun entraînement de modèle n'est effectué ici : le modèle entraîné est fourni.

Usages :
  python app/interface_clinique.py                 -> menu interactif
  python app/interface_clinique.py --demo          -> exemples automatiques
  python app/interface_clinique.py --modele neurones
"""

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from core.clinical_predictor import ClinicalPredictor
from pipeline.trainer import entrainer_depuis_csv, construire_modele

DESCRIPTIONS = {
    "age": ("Âge du patient (années)", "45"),
    "temperature": ("Température (°C)", "38.5"),
    "frequence_cardiaque": ("Fréquence cardiaque (bpm)", "90"),
    "test_serologique": ("Test sérologique (0 = négatif, 1 = positif)", "1"),
}


def construire_predicteur(nom_modele="logistique", verbose=True):
    """Entraîne un modèle puis le confie à un ClinicalPredictor."""
    chemin = os.path.join(RACINE, "data", "patient_data.csv")
    modele, dataset, _, _ = entrainer_depuis_csv(chemin, nom_modele, verbose=verbose)
    return ClinicalPredictor(modele), dataset


def afficher_diagnostic(predictor, dataset, patient_brut):
    """Normalise le patient, prédit et affiche un diagnostic lisible."""
    vecteur = dataset.normaliser_patient(patient_brut)
    proba = predictor.predict_proba(vecteur)
    diagnostic = predictor.diagnose(vecteur)

    details = ", ".join(f"{nom}={valeur}" for nom, valeur in
                        zip(dataset.noms_features, patient_brut))
    indicateur = "SSS" if diagnostic == "Infecté" else "---"
    print(f"\n  Patient : {details}")
    print(f"  Probabilité d'infection : {proba:.4f}")
    print(f"  Diagnostic : [{indicateur}] {diagnostic}")


def saisir_patient(dataset):
    """Demande les features du patient au médecin (valeur par défaut possible)."""
    print("\nSaisir les valeurs (Entrée = valeur par défaut) :")
    valeurs = []
    for nom in dataset.noms_features:
        libelle, defaut = DESCRIPTIONS.get(nom, (nom, "0"))
        reponse = input(f"  {libelle} [{defaut}] : ").strip()
        valeurs.append(float(reponse) if reponse else float(defaut))
    return valeurs


def exemples(predictor, dataset):
    """Affiche quelques diagnostics de démonstration."""
    cas = [
        [45, 39.2, 105, 1],   # forte fièvre + test positif
        [22, 36.8, 72, 0],    # aucun signe
        [70, 38.9, 98, 0],    # personne âgée, fièvre, test négatif
        [35, 37.1, 80, 1],    # test positif isolé
    ]
    print("\n=== Exemples de démonstration ===")
    for patient in cas:
        afficher_diagnostic(predictor, dataset, patient)


def menu(predictor, dataset):
    """Boucle du menu interactif."""
    while True:
        print("\n" + "=" * 48)
        print("  INTERFACE CLINIQUE — Diagnostic viral")
        print("=" * 48)
        print("  1. Diagnostiquer un nouveau patient")
        print("  2. Voir des exemples de démonstration")
        print("  3. Quitter")
        choix = input("Votre choix : ").strip()

        if choix == "1":
            patient = saisir_patient(dataset)
            afficher_diagnostic(predictor, dataset, patient)
        elif choix == "2":
            exemples(predictor, dataset)
        elif choix == "3":
            print("\nFin de la consultation. Bonne journée.")
            break
        else:
            print("  Choix invalide, veuillez réessayer.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    nom_modele = "neurones" if "--modele" in sys.argv and "neurones" in sys.argv else "logistique"
    predictor, dataset = construire_predicteur(nom_modele)
    print(f"\nModèle chargé : « {predictor.model.nom} »")
    print(dataset.resume())

    if "--demo" in sys.argv:
        exemples(predictor, dataset)
    else:
        menu(predictor, dataset)


if __name__ == "__main__":
    main()
