# 🎓 Système de surveillance anti-triche pour examens pratiques

Ce projet est un prototype local développé en Python dans le cadre d’un PFA. Il permet de surveiller un étudiant pendant un examen pratique sur poste informatique, à l’aide de la webcam, sans bloquer l’utilisation normale de l’ordinateur.

L’étudiant peut donc continuer à travailler normalement sur son énoncé PDF et dans son environnement de développement, par exemple VS Code, pendant que l’application détecte automatiquement certaines situations suspectes.

---

## 🎯 Objectif du projet

L’objectif est de proposer un outil simple d’aide à la surveillance pendant les examens pratiques. Le système ne remplace pas le professeur et ne prend aucune décision de sanction. Il enregistre seulement des alertes et des preuves visuelles que l’enseignant peut consulter après l’examen.

Le prototype permet principalement de détecter :

- la présence d’un téléphone portable dans le champ de la caméra ;
- la présence de plusieurs personnes devant la webcam ;
- un regard ou une orientation de tête hors écran pendant une durée prolongée ;
- les événements suspects avec horodatage, niveau de gravité et capture d’image.

---

## ✅ Fonctionnalités principales

- Interface graphique locale avec Tkinter.
- Saisie du matricule étudiant.
- Vérification du couple matricule / nom de l’étudiant.
- Choix de la matière depuis une liste prédéfinie.
- Durée d’examen attribuée automatiquement selon la matière.
- Blocage d’une matière déjà utilisée par le même étudiant.
- Vérification de l’accès à la webcam avant la surveillance.
- Détection de personnes et de téléphones avec YOLOv8.
- Détection approximative du regard hors écran avec MediaPipe.
- Affichage en direct du nombre d’alertes, du nombre de personnes détectées et du nombre de téléphones détectés.
- Enregistrement des alertes dans un fichier `alerts.jsonl`.
- Capture d’image automatique lorsqu’une alerte est déclenchée.
- Génération d’un résumé final dans `summary.json`.
- Arrêt manuel avec la touche `q`.
- Arrêt automatique lorsque la durée de l’examen est atteinte.

---

## 🧠 Types d’alertes gérés

| Type d’alerte | Description | Gravité |
|---|---|---|
| `phone_detected` | Un téléphone portable est détecté par la webcam | Critical |
| `multiple_people_detected` | Deux personnes ou plus sont visibles devant la caméra | High |
| `offscreen_gaze` | Le candidat regarde hors écran ou tourne la tête trop longtemps | Medium |

---

## 🛠️ Technologies utilisées

- **Python 3.11** : langage principal du projet.
- **Tkinter** : interface graphique locale.
- **OpenCV** : accès webcam, affichage vidéo et sauvegarde des captures.
- **YOLOv8 / Ultralytics** : détection d’objets (`person`, `cell phone`).
- **MediaPipe** : estimation de l’orientation du visage et du regard.
- **JSON / JSONL** : stockage local des alertes et du résumé de session.

---

## 📁 Structure du projet

```txt
CheatingDetector/
│
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration d’une session
│   ├── detectors.py       # Détection YOLO + MediaPipe
│   ├── gui.py             # Interface graphique Tkinter
│   ├── logger.py          # Journalisation des alertes et résumé
│   ├── session.py         # Boucle principale de surveillance
│   ├── students.py        # Liste des étudiants et validation matricule/nom
│   ├── subjects.py        # Liste des matières et durées
│   └── used_pairs.py      # Gestion des matières déjà passées
│
├── main.py                # Point d’entrée de l’application
├── requirements.txt       # Dépendances Python
├── students.xlsx          # Fichier source des étudiants
├── README.md
└── .gitignore
```

---

## ⚙️ Prérequis

Avant de lancer le projet, il faut avoir :

- Windows ;
- Python 3.11 ;
- une webcam fonctionnelle ;
- Git, si le projet est récupéré depuis GitHub ;
- le modèle YOLO `yolov8n.pt` placé à la racine du projet.

> Remarque : le fichier `yolov8n.pt` n’est pas inclus dans GitHub car il est ignoré par `.gitignore`.

---

## 📥 Installation

### 1. Cloner le dépôt

```powershell
git clone https://github.com/Amin55-rr/CheatingDetector.git
cd CheatingDetector
```

### 2. Créer un environnement virtuel

```powershell
py -3.11 -m venv .venv311
```

### 3. Activer l’environnement virtuel

```powershell
.\.venv311\Scripts\activate
```

### 4. Installer les dépendances

```powershell
pip install -r requirements.txt
```

## ▶️ Lancement du projet

Après installation, lancer l’application avec :

```powershell
python main.py
```

L’interface graphique s’ouvre ensuite automatiquement.

---

## 🧾 Utilisation de l’application

1. Saisir les 6 derniers chiffres du matricule étudiant.

   Exemple :

   ```txt
   000011
   ```

   Le programme complète automatiquement le préfixe :

   ```txt
   C-2026-000011
   ```

2. Choisir le nom et prénom de l’étudiant dans la liste.

3. Choisir la matière disponible.

4. Vérifier la durée affichée automatiquement.

5. Cliquer sur :

   ```txt
   Démarrer la surveillance
   ```

6. Pendant la session, une fenêtre webcam s’ouvre.

7. Pour arrêter manuellement la surveillance, appuyer sur :

   ```txt
   q
   ```

8. Si la durée de l’examen est atteinte, le programme s’arrête automatiquement.

---

## ⏱️ Durée des matières

Les durées sont définies automatiquement dans `app/subjects.py`.

| Matière                           | Durée  |
|-----------------------------------|-------:|
| Communication Professionnelle 2   |   1h   |
| Intermediate English 2            |   1h   |
| Conception Orientée Objet         |   2h   |
| Modèles Statistiques              |   2h   |
| Programmation Linéaire            |   2h   |
| Programmation Python et Framework |   3h   |
| Programmation Java                |   3h   |
| SQL Server                        |   3h   |
| Recherche Scientifique            |   3h   |
| Réseaux Informatiques 2           |   3h   |

---

## 🔐 Gestion des matières déjà passées

Le projet empêche un même étudiant de repasser la même matière.

Après une session terminée, le couple suivant est enregistré localement :

```txt
matricule + matière
```

Le fichier utilisé est :

```txt
data/used_student_subjects.json
```

Ainsi, si l’étudiant sélectionne de nouveau la même matière, elle ne sera plus disponible dans l’interface.

---

## 📊 Données générées après une session

Chaque session crée automatiquement un dossier dans :

```txt
sessions/
```

Le nom du dossier suit ce format :

```txt
sessions/<student_id>_<date_heure>/
```

Exemple :

```txt
sessions/C-2026-000011_20260429_221500/
```

Ce dossier contient :

```txt
alerts.jsonl
summary.json
captures/
```

### `alerts.jsonl`

Contient le journal brut des alertes. Chaque ligne correspond à une alerte avec :

- la date et l’heure ;
- le type d’alerte ;
- la gravité ;
- les informations complémentaires ;
- le chemin de la capture associée si disponible.

### `summary.json`

Contient le résumé final de la session :

- informations de session ;
- nombre total d’alertes ;
- répartition des alertes par type ;
- liste complète des événements.

### `captures/`

Contient les images enregistrées automatiquement lors des alertes.

---

## 🧩 Description rapide des modules

### `main.py`

Point d’entrée du programme. Il lance l’interface graphique.

### `app/gui.py`

Gère l’interface utilisateur : saisie du matricule, choix de l’étudiant, choix de la matière, affichage de la durée et démarrage de la session.

### `app/session.py`

Contient la boucle principale de surveillance. Ce module lit les images de la webcam, appelle le détecteur, déclenche les alertes, affiche les informations à l’écran et arrête la session avec `q` ou à la fin du temps.

### `app/detectors.py`

Analyse chaque image webcam. Il utilise YOLOv8 pour détecter les personnes et téléphones, puis MediaPipe pour estimer si le regard ou la tête est hors écran.

### `app/logger.py`

Enregistre les alertes dans `alerts.jsonl`, sauvegarde les captures et génère `summary.json`.

### `app/students.py`

Contient la liste des étudiants et vérifie que le matricule correspond bien au nom choisi.

### `app/subjects.py`

Contient la liste des matières et leurs durées en minutes.

### `app/used_pairs.py`

Stocke les couples étudiant / matière déjà utilisés pour empêcher un étudiant de repasser la même matière.

### `app/config.py`

Définit les paramètres de session : webcam, seuil de confiance, durée avant alerte, cooldown entre alertes, etc.

---

## 🚫 Fichiers à ne pas pousser sur GitHub

Les fichiers suivants doivent rester ignorés par Git :

```txt
.venv/
.venv311/
build/
dist/
sessions/
data/
*.spec
yolov8n.pt
__pycache__/
*.pyc
```

Ces fichiers sont soit générés automatiquement, soit liés à l’environnement local.

---

## ⚠️ Limites du projet

- Le système est un prototype académique, pas une solution industrielle.
- La détection dépend fortement de la qualité de la webcam, de l’éclairage et de la position de l’étudiant.
- La détection du regard est approximative et ne constitue pas une preuve définitive de fraude.
- Les faux positifs restent possibles, notamment avec la détection d’objets.
- Le programme ne bloque pas le système d’exploitation, Internet, le clavier ou les autres applications.
- Le système ne prend aucune décision automatique de sanction.

---

## 🎓 Conclusion

Ce projet propose une solution locale, simple et réaliste pour assister la surveillance des examens pratiques. Il permet de détecter automatiquement des comportements suspects visibles à la webcam, d’enregistrer les alertes et de produire un résumé exploitable par l’enseignant après l’épreuve.

L’application reste compatible avec le déroulement habituel d’un examen pratique : l’étudiant consulte son énoncé PDF, travaille dans VS Code, et le système fonctionne en parallèle comme outil d’aide à la surveillance.
