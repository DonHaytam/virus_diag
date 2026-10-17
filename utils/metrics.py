"""utils/metrics.py — Calcul des scores : exactitude, précision, rappel, F1.

Ce sont des FONCTIONS UTILITAIRES : aucun modèle d'IA ici.
"""

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)


def matrice_confusion(y_vrai, y_pred):
    """Renvoie {'vp', 'vn', 'fp', 'fn'} (vrais/faux positifs et négatifs)."""
    vp = vn = fp = fn = 0
    for vrai, predit in zip(y_vrai, y_pred):
        if vrai == 1 and predit == 1:
            vp += 1
        elif vrai == 0 and predit == 0:
            vn += 1
        elif vrai == 0 and predit == 1:
            fp += 1
        else:
            fn += 1
    return {"vp": vp, "vn": vn, "fp": fp, "fn": fn}


def exactitude(y_vrai, y_pred):
    """Proportion de diagnostics corrects (accuracy)."""
    m = matrice_confusion(y_vrai, y_pred)
    total = m["vp"] + m["vn"] + m["fp"] + m["fn"]
    return (m["vp"] + m["vn"]) / total if total else 0.0


def precision(y_vrai, y_pred):
    """Parmi les patients déclarés infectés, ceux réellement infectés."""
    m = matrice_confusion(y_vrai, y_pred)
    denominateur = m["vp"] + m["fp"]
    return m["vp"] / denominateur if denominateur else 0.0


def rappel(y_vrai, y_pred):
    """Parmi les patients réellement infectés, ceux détectés (sensibilité)."""
    m = matrice_confusion(y_vrai, y_pred)
    denominateur = m["vp"] + m["fn"]
    return m["vp"] / denominateur if denominateur else 0.0


def f1(y_vrai, y_pred):
    """Moyenne harmonique de la précision et du rappel."""
    p = precision(y_vrai, y_pred)
    r = rappel(y_vrai, y_pred)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def rapport_classification(y_vrai, y_pred, titre="Rapport de classification"):
    """Construit un rapport lisible (matrice de confusion + scores)."""
    m = matrice_confusion(y_vrai, y_pred)
    lignes = [
        titre,
        "-" * len(titre),
        f"  Matrice de confusion : VP={m['vp']}  VN={m['vn']}  FP={m['fp']}  FN={m['fn']}",
        f"  Exactitude : {exactitude(y_vrai, y_pred):.4f}",
        f"  Précision  : {precision(y_vrai, y_pred):.4f}",
        f"  Rappel     : {rappel(y_vrai, y_pred):.4f}",
        f"  F1-score   : {f1(y_vrai, y_pred):.4f}",
    ]
    return "\n".join(lignes)
