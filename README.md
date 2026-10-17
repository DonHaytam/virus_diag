# virus_diag — Diagnostic viral (Exercice POO, Chapitre 5)

Projet d'entraînement d'un modèle IA qui diagnostique une infection virale à
partir de données patient **artificielles**. Objectif pédagogique : illustrer la
**Programmation Orientée Objet** et une **architecture IA séparée en couches**.

## Arborescence

```
virus_diag/
├── data/
│   └── patient_data.csv          # 693 patients générés (features + cible)
├── pipeline/
│   ├── trainer.py                # Trainer : entraîne le modèle
│   └── evaluator.py              # Evaluator : mesure les performances
├── utils/
│   ├── preprocessing.py          # nettoyage / normalisation des données
│   └── metrics.py                # exactitude, précision, rappel, F1
├── core/
│   ├── dataset.py                # chargement / représentation des données
│   ├── model.py                  # classe de base ModeleIA (contrat commun)
│   ├── logistic_regression.py    # régression logistique
│   ├── neural_network.py         # réseau de neurones (1 couche cachée)
│   ├── optimizer.py              # descente de gradient (avec momentum)
│   └── clinical_predictor.py     # ClinicalPredictor : diagnostic (prédiction)
└── app/
    ├── api.py                    # API (bibliothèque + serveur HTTP)
    └── interface_clinique.py     # interface pour les médecins
```

Logique générale :

```
Données (data) → Prétraitement (utils) → Dataset & modèles (core)
      → Entraînement & évaluation (pipeline) → Application (app)
```

## Rôle de chaque dossier

| Dossier / fichier | Rôle |
|---|---|
| `data/` | Les données du projet. |
| `patient_data.csv` | Patients utilisés pour l'entraînement et le test. |
| `core/` | Composants fondamentaux de l'IA (modèles, dataset, optimiseur, diagnostic). |
| `core/model.py` | Classe de base `ModeleIA` : contrat commun (`entrainer`, `predict`). |
| `core/logistic_regression.py` | Implémentation de la régression logistique. |
| `core/neural_network.py` | Implémentation du réseau de neurones. |
| `core/optimizer.py` | Optimisation des paramètres (descente de gradient). |
| `core/clinical_predictor.py` | Diagnostic à partir d'un modèle **déjà entraîné**. |
| `pipeline/` | Organise l'entraînement et l'évaluation. |
| `pipeline/trainer.py` | Entraîne le modèle. |
| `pipeline/evaluator.py` | Évalue les performances. |
| `utils/` | Fonctions utilitaires (aucun modèle d'IA). |
| `utils/preprocessing.py` | Nettoie et normalise les données. |
| `utils/metrics.py` | Calcule précision, rappel, F1, exactitude. |
| `app/` | Interfaces d'utilisation du système. |
| `app/api.py` | Expose le diagnostic via une API. |
| `app/interface_clinique.py` | Interface destinée aux médecins. |

## Données artificielles

`data/patient_data.csv` est **généré** (graine fixe, reproductible) avec les
colonnes :

| Colonne | Description |
|---|---|
| `age` | Âge (années) |
| `temperature` | Température (°C) |
| `frequence_cardiaque` | Fréquence cardiaque (bpm) |
| `test_serologique` | 0 = négatif, 1 = positif |
| `infection` | Cible : 1 = infecté, 0 = sain |

Le risque d'infection augmente avec l'âge et les signes cliniques ; ~2 % de
bruit est ajouté sur l'étiquette pour rendre le problème réaliste.

## Exécution (Python 3, **sans dépendance externe**)

Depuis le dossier `virus_diag/` :

```bash
python core/dataset.py                     # aperçu du jeu de données
python pipeline/trainer.py                 # entraîne les deux modèles
python pipeline/evaluator.py               # métriques + comparaison
python core/clinical_predictor.py          # démo polymorphisme
python app/interface_clinique.py --demo    # interface clinique (exemples)
python app/interface_clinique.py           # interface clinique (menu)
python app/api.py                          # API (exemples JSON)
python app/api.py --serve                  # API en serveur HTTP (port 8000)
```

> Sur Windows, pour un affichage correct des accents : `set PYTHONIOENCODING=utf-8`.

Exemple de résultats obtenus (jeu de test, seuil 0.5) :

| Modèle | Exactitude | Précision | Rappel | F1 |
|---|---|---|---|---|
| Régression logistique | 0.9353 | 0.8478 | 0.9512 | 0.8966 |
| Réseau de neurones | 0.9856 | 0.9535 | 1.0000 | 0.9762 |

## Exercice — réponses

### 1. La classe `ClinicalPredictor`

Fichier : `core/clinical_predictor.py`

- reçoit un **modèle déjà entraîné** via le constructeur ;
- expose `diagnose(patient_data)` ;
- renvoie `"Infecté"` si la prédiction **≥ 0.5**, sinon `"Sain"`.

```python
class ClinicalPredictor:
    SEUIL_DECISION = 0.5

    def __init__(self, model, seuil=SEUIL_DECISION):
        if not callable(getattr(model, "predict", None)):
            raise TypeError("Le modèle fourni doit implémenter une méthode predict(patient_data).")
        self.model = model          # modèle reçu DÉJÀ entraîné
        self.seuil = seuil

    def diagnose(self, patient_data):
        if self.predict_proba(patient_data) >= self.seuil:
            return "Infecté"
        return "Sain"
```

### 2. Quel concept POO ?

Le **polymorphisme**, et plus précisément le **duck typing** en Python :
`ClinicalPredictor` n'impose aucun type précis, il exige seulement une méthode
`predict()`. La régression logistique, le réseau de neurones, ou n'importe quel
modèle externe (sklearn, TensorFlow…) fonctionnent donc avec la **même** classe,
sans la modifier. La classe de base `ModeleIA` (`core/model.py`) matérialise ce
contrat commun.

### 3. Pourquoi séparer la prédiction (`ClinicalPredictor`) de l'entraînement (`Trainer`) ?

- **Responsabilité unique (SRP)** : `Trainer` entraîne (phase lourde et rare) ;
  `ClinicalPredictor` diagnostique (usage léger et fréquent).
- **Cycles de vie différents** : on entraîne une fois, on prédit des milliers de fois.
- **Sécurité clinique** : l'interface (`app/`) ne peut pas modifier les poids du modèle.
- **Testabilité** : `ClinicalPredictor` se teste avec un faux modèle, sans données ni ré-entraînement.
- **Réutilisabilité** : le même `ClinicalPredictor` sert l'API et l'interface clinique.
- **Déploiement** : en production, on déploie le modèle sérialisé + le wrapper,
  pas le pipeline d'entraînement.
