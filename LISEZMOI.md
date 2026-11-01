# Transcripteur de réunions

Upload un audio (jusqu'à 2 h et plus) → transcription horodatée + compte rendu (résumé, décisions, actions) + glossaire des termes techniques.

## Installation (une seule fois)

1. Crée une clé API gratuite (sans carte bancaire) : https://console.groq.com/keys
2. Dans un dossier contenant `app.py`, `requirements.txt` et le dossier `.streamlit/` :

```bash
pip install -r requirements.txt
```

ffmpeg est fourni automatiquement par `imageio-ffmpeg`, rien d'autre à installer.

## Lancement

```bash
streamlit run app.py
```

Le navigateur s'ouvre sur http://localhost:8501. Colle ta clé dans le panneau de gauche (ou définis la variable d'environnement `GROQ_API_KEY` pour ne plus la retaper).

## Astuces

- Remplis le champ **Vocabulaire technique** avec les noms d'outils, acronymes et noms de collègues : Whisper les écrira correctement.
- Palier gratuit Groq : ~2 h d'audio par heure et ~8 h par jour. Si la limite est atteinte, l'app patiente et relance seule.
- Formats acceptés : mp3, m4a, wav, ogg, webm, mp4, flac…
- La transcription ne distingue pas qui parle (pas de séparation des intervenants).
