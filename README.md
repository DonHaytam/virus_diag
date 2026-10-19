
# virus_diag — Diagnostic viral (Exercice POO, Chapitre 5)

Projet d'entraînement d'un modèle IA qui diagnostique une infection virale à
partir de données patient **artificielles**. Objectif pédagogique : illustrer la
**Programmation Orientée Objet** et une **architecture IA séparée en couches**.




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
