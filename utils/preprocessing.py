"""utils/preprocessing.py — Nettoyage et préparation des données.

Ce sont des FONCTIONS UTILITAIRES : elles n'entraînent AUCUN modèle.
Elles préparent les données avant de les donner au `core` puis au `pipeline`.

Place dans l'arborescence : data -> utils -> core -> pipeline -> app
"""

import csv
import os
import random
import sys

# --- Bootstrap : permet d'exécuter ce fichier directement depuis n'importe où
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)


def nettoyer_lignes(entetes, lignes):
    """Nettoyage simple : retire les lignes incomplètes, vides ou en double."""
    vues = set()
    propres = []
    for ligne in lignes:
        if len(ligne) != len(entetes):          # ligne mal formée
            continue
        if any(cellule.strip() == "" for cellule in ligne):  # valeur manquante
            continue
        cle = tuple(cellule.strip() for cellule in ligne)
        if cle in vues:                          # doublon
            continue
        vues.add(cle)
        propres.append([cellule.strip() for cellule in ligne])
    return propres


def charger_csv(chemin):
    """Charge un CSV et renvoie (entetes, lignes) après nettoyage."""
    with open(chemin, newline="", encoding="utf-8") as fichier:
        lecteur = csv.reader(fichier)
        entetes = [entete.strip() for entete in next(lecteur)]
        lignes = [ligne for ligne in lecteur if any(cellule.strip() for cellule in ligne)]
    return entetes, nettoyer_lignes(entetes, lignes)


def extraire_xy(chemin, colonne_cible="infection"):
    """Sépare les features (X) de la cible (y).

    Renvoie (X, y, noms_features) où X est une liste de listes de floats.
    """
    entetes, lignes = charger_csv(chemin)
    if colonne_cible not in entetes:
        raise ValueError(f"Colonne cible '{colonne_cible}' absente de {chemin}")
    index_cible = entetes.index(colonne_cible)
    noms_features = [nom for i, nom in enumerate(entetes) if i != index_cible]

    X, y = [], []
    for ligne in lignes:
        features = [float(valeur) for i, valeur in enumerate(ligne) if i != index_cible]
        X.append(features)
        y.append(int(float(ligne[index_cible])))
    return X, y, noms_features


def statistiques_minmax(X):
    """Calcule (minimum, maximum) de chaque colonne de X."""
    n_colonnes = len(X[0])
    stats = []
    for j in range(n_colonnes):
        colonne = [ligne[j] for ligne in X]
        stats.append((min(colonne), max(colonne)))
    return stats


def appliquer_normalisation(X, stats):
    """Applique une normalisation min-max déjà calculée (bornes -> [0, 1])."""
    X_norm = []
    for ligne in X:
        sortie = []
        for j, valeur in enumerate(ligne):
            minimum, maximum = stats[j]
            etendue = (maximum - minimum) if maximum > minimum else 1.0
            sortie.append((valeur - minimum) / etendue)
        X_norm.append(sortie)
    return X_norm


def normaliser(X, stats=None):
    """Normalise (min-max) X dans [0, 1]. Renvoie (X_normalise, stats)."""
    if stats is None:
        stats = statistiques_minmax(X)
    return appliquer_normalisation(X, stats), stats


def separer_train_test(X, y, ratio_test=0.2, seed=42):
    """Sépare (X, y) en jeu d'entraînement et jeu de test (mélange reproductible)."""
    rng = random.Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    n_test = max(1, int(round(len(X) * ratio_test)))
    idx_test = indices[:n_test]
    idx_train = indices[n_test:]

    X_train = [X[i] for i in idx_train]
    y_train = [y[i] for i in idx_train]
    X_test = [X[i] for i in idx_test]
    y_test = [y[i] for i in idx_test]
    return X_train, X_test, y_train, y_test
