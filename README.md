# Prototype de surveillance anti-triche (PFA)

Application locale Python pour surveiller un examen pratique via webcam pendant que l'etudiant travaille dans VS Code.

## Fonctionnalites MVP

- Demarrage de session (etudiant + matiere choisie dans une liste).
- Duree d'examen fixee automatiquement selon la matiere.
- Blocage: un meme identifiant etudiant ne peut pas repasser la meme matiere (stockage local).
- Verification webcam.
- Detection `person` et `cell phone` via YOLOv8.
- Detection simple d'orientation de tete (approximation regard hors ecran) via MediaPipe.
- Alertes horodatees en `alerts.jsonl`.
- Captures images en cas d'alerte.
- Resume de session en `summary.json`.

## Prerequis

- Python 3.11 recommande pour cet environnement Windows.
- Webcam fonctionnelle.

## Installation

```bash
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install -r requirements.txt
```

## Execution

```bash
.venv311\Scripts\python main.py
```

Lancement via interface graphique:

- L'etudiant remplit le formulaire (code, matiere, seuil regard hors ecran).
- Cliquer `Demarrer la surveillance`.
- Pendant la surveillance, appuyer sur `q` dans la fenetre webcam pour arreter manuellement.

## Donnees locales (hors sessions)

Le fichier suivant memorise les couples (etudiant, matiere) deja utilises:

`data/used_student_subjects.json`

## Structure de sortie

Chaque session cree un dossier:

`sessions/<student_id>_<timestamp>/`

avec:

- `alerts.jsonl`: journal brut des alertes
- `summary.json`: resume final de la session
- `captures/`: preuves image horodatees

## Limites (version actuelle)

- Le regard hors ecran est une estimation simplifiee (MediaPipe si disponible, sinon fallback OpenCV), pas un eye-tracking precis
- Les performances dependent de la webcam et de l'eclairage
- Ce prototype aide la surveillance, il ne prend aucune decision de sanction
