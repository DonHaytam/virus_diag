"""app/api.py — Expose le diagnostic via une API.

C'est la couche « application » : elle ne contient ni entraînement ni modèle.
Elle s'appuie sur un ClinicalPredictor déjà construit par le pipeline.

Deux usages :
  - en tant que bibliothèque : APIClinique.diagnostiquer({...})
  - en tant que serveur HTTP : python app/api.py --serve
"""

import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from core.clinical_predictor import ClinicalPredictor
from pipeline.trainer import entrainer_depuis_csv


class APIClinique:
    """API applicative : transforme une requête patient en réponse JSON."""

    def __init__(self, predictor, dataset):
        self.predictor = predictor
        self.dataset = dataset

    def sante(self):
        """Point de contrôle de l'API (metadonnées du service)."""
        return {
            "statut": "ok",
            "modele": self.predictor.model.nom,
            "features_attendues": self.dataset.noms_features,
            "seuil_decision": self.predictor.seuil,
        }

    def _vers_vecteur(self, patient):
        """Convertit un dictionnaire patient en vecteur normalisé, dans l'ordre."""
        vecteur_brut = []
        manquants = []
        for nom in self.dataset.noms_features:
            if nom not in patient:
                manquants.append(nom)
            else:
                vecteur_brut.append(float(patient[nom]))
        if manquants:
            raise KeyError("Champs manquants : " + ", ".join(manquants))
        return self.dataset.normaliser_patient(vecteur_brut)

    def diagnostiquer(self, patient):
        """Requête principale : renvoie le diagnostic et la probabilité."""
        vecteur = self._vers_vecteur(patient)
        proba = self.predictor.predict_proba(vecteur)
        return {
            "diagnostic": self.predictor.diagnose(vecteur),
            "probabilite_infection": round(proba, 4),
            "seuil_decision": self.predictor.seuil,
        }


def construire_api(nom_modele="logistique", verbose=False):
    """Entraîne le modèle puis construit une API prête à l'emploi."""
    chemin = os.path.join(RACINE, "data", "patient_data.csv")
    modele, dataset, _, _ = entrainer_depuis_csv(chemin, nom_modele, verbose=verbose)
    return APIClinique(ClinicalPredictor(modele), dataset)


def lancer_serveur(api, hote="127.0.0.1", port=8000):
    """Petit serveur HTTP (stdlib) : GET /sante et POST /diagnostic."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _envoyer(self, code, charge):
            corps = json.dumps(charge, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)

        def do_GET(self):
            if self.path == "/sante":
                self._envoyer(200, api.sante())
            else:
                self._envoyer(404, {"erreur": "route inconnue"})

        def do_POST(self):
            if self.path != "/diagnostic":
                self._envoyer(404, {"erreur": "route inconnue"})
                return
            longueur = int(self.headers.get("Content-Length", 0))
            try:
                patient = json.loads(self.rfile.read(longueur) or b"{}")
                self._envoyer(200, api.diagnostiquer(patient))
            except (ValueError, KeyError, TypeError) as erreur:
                self._envoyer(400, {"erreur": str(erreur)})

        def log_message(self, *args):
            pass  # silencieux

    print(f"API en écoute sur http://{hote}:{port}")
    print("  GET  /sante       -> état du service")
    print("  POST /diagnostic  -> {'age':..,'temperature':..,'frequence_cardiaque':..,'test_serologique':..}")
    print("Ctrl+C pour arrêter.")
    HTTPServer((hote, port), Handler).serve_forever()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    api = construire_api("logistique")

    print("=== GET /sante ===")
    print(json.dumps(api.sante(), indent=2, ensure_ascii=False))

    exemples = [
        {"age": 45, "temperature": 39.2, "frequence_cardiaque": 105, "test_serologique": 1},
        {"age": 22, "temperature": 36.8, "frequence_cardiaque": 72, "test_serologique": 0},
    ]
    for patient in exemples:
        print(f"\n=== POST /diagnostic  {patient} ===")
        print(json.dumps(api.diagnostiquer(patient), indent=2, ensure_ascii=False))

    if "--serve" in sys.argv:
        lancer_serveur(api, port=int(os.environ.get("PORT", 8000)))
