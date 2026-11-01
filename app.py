"""
Transcripteur de réunions — Streamlit + API Groq (palier gratuit)

- Transcription : Whisper Large v3 (Groq)
- Résumé + glossaire des termes techniques : GPT-OSS 120B (Groq)
- Les longs audios (1h, 2h...) sont découpés automatiquement en morceaux de 10 min.

Lancement :  streamlit run app.py
"""

import os
import re
import tempfile
import subprocess
import time
from pathlib import Path

import requests
import streamlit as st
import imageio_ffmpeg

API = "https://api.groq.com/openai/v1"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # ffmpeg embarqué, pas d'install système
CHUNK_SEC = 600  # 10 min par morceau (~2,4 Mo en mp3 32 kbps, bien sous la limite de 25 Mo)


# ---------------------------------------------------------------- utilitaires
def hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def duree_audio(path: str) -> float:
    """Durée en secondes, lue dans la sortie de ffmpeg."""
    out = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True, errors="ignore").stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
    if not m:
        raise ValueError("Impossible de lire ce fichier audio.")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def decouper(path: str, dossier: str) -> list[str]:
    """Convertit en mp3 mono 16 kHz et découpe en morceaux de CHUNK_SEC secondes."""
    motif = os.path.join(dossier, "part_%03d.mp3")
    cmd = [
        FFMPEG, "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
        "-f", "segment", "-segment_time", str(CHUNK_SEC), "-reset_timestamps", "1", motif,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    if r.returncode != 0:
        raise RuntimeError("Erreur ffmpeg :\n" + r.stderr[-1500:])
    return sorted(str(p) for p in Path(dossier).glob("part_*.mp3"))


def post_groq(url: str, cle: str, statut=None, **kwargs) -> dict:
    """POST avec relance automatique si on touche la limite du palier gratuit (HTTP 429)."""
    headers = {"Authorization": f"Bearer {cle}"}
    for tentative in range(8):
        r = requests.post(url, headers=headers, timeout=300, **kwargs)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            attente = float(r.headers.get("retry-after", 0) or 0) or min(15 * (tentative + 1), 90)
            if statut:
                statut.info(f"Limite gratuite atteinte, nouvelle tentative dans {int(attente)} s…")
            time.sleep(attente)
            continue
        if r.status_code == 401:
            raise RuntimeError("Clé API Groq invalide.")
        raise RuntimeError(f"Erreur API ({r.status_code}) : {r.text[:500]}")
    raise RuntimeError("Trop de tentatives : limite gratuite saturée, réessaie plus tard.")


def transcrire(chemin, cle, modele, langue, vocab, statut) -> list[dict]:
    with open(chemin, "rb") as f:
        contenu = f.read()
    data = {"model": modele, "response_format": "verbose_json", "temperature": "0"}
    if langue:
        data["language"] = langue
    if vocab:
        # Le "prompt" aide Whisper à bien orthographier les termes techniques
        data["prompt"] = vocab[:600]
    res = post_groq(
        f"{API}/audio/transcriptions", cle, statut,
        data=data, files={"file": (os.path.basename(chemin), contenu, "audio/mpeg")},
    )
    segs = res.get("segments") or [{"start": 0, "text": res.get("text", "")}]
    return [{"start": s.get("start", 0), "text": s.get("text", "").strip()} for s in segs]


def chat(cle, modele, systeme, contenu, statut, max_tokens=3000) -> str:
    res = post_groq(
        f"{API}/chat/completions", cle, statut,
        json={
            "model": modele,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "reasoning_effort": "low",  # modèles gpt-oss : limite la réflexion interne
            "messages": [{"role": "system", "content": systeme}, {"role": "user", "content": contenu}],
        },
    )
    texte = (res["choices"][0]["message"].get("content") or "").strip()
    if not texte:
        raise RuntimeError("Le modèle a renvoyé une réponse vide, réessaie (ou choisis l'autre modèle).")
    return texte


SYS_NOTES = (
    "Tu es un assistant qui prend des notes de réunion professionnelles en français. "
    "On te donne un extrait de transcription automatique (il peut contenir des erreurs de reconnaissance). "
    "Produis des notes concises en Markdown avec : sujets abordés, décisions, actions (qui / quoi / quand si mentionné), "
    "questions ouvertes, et la liste des termes techniques, acronymes ou jargon entendus. "
    "N'invente rien qui ne soit pas dans l'extrait."
)

SYS_FINAL = (
    "Tu es un assistant qui rédige des comptes rendus de réunion clairs, en français. "
    "On te donne des notes prises sur les parties successives d'une réunion. Rédige le compte rendu final en Markdown "
    "avec exactement ces sections :\n"
    "## Résumé (5 à 10 lignes)\n## Points clés\n## Décisions\n## Actions à faire (qui / quoi / quand)\n"
    "## Questions ouvertes\n## Glossaire des termes techniques\n"
    "Dans le glossaire, explique chaque terme technique ou acronyme simplement, en une ou deux phrases, "
    "comme à un alternant qui découvre le sujet. Si un acronyme est propre à l'entreprise et que tu ne connais pas "
    "son sens avec certitude, propose une hypothèse et ajoute « (à vérifier) ». "
    "N'invente pas de décisions ni d'actions absentes des notes."
)


# ---------------------------------------------------------------- interface
st.set_page_config(page_title="Transcripteur de réunions", page_icon="🎙️", layout="wide")


def secret(nom: str) -> str:
    """Lit un secret (Streamlit Cloud / .streamlit/secrets.toml), sinon une variable d'environnement."""
    try:
        if nom in st.secrets:
            return str(st.secrets[nom])
    except Exception:
        pass
    return os.environ.get(nom, "")


# --- Protection par mot de passe (si APP_PASSWORD est défini dans les secrets)
MOT_DE_PASSE = secret("APP_PASSWORD")
if MOT_DE_PASSE and not st.session_state.get("connecte"):
    st.title("🔒 Transcripteur de réunions")
    mdp = st.text_input("Mot de passe", type="password")
    if mdp:
        if mdp == MOT_DE_PASSE:
            st.session_state["connecte"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()

CLE_SERVEUR = secret("GROQ_API_KEY")

st.title("🎙️ Transcripteur de réunions")
st.caption("Upload un enregistrement → transcription horodatée + compte rendu + glossaire des termes techniques.")

with st.sidebar:
    st.header("Réglages")
    if CLE_SERVEUR:
        cle = CLE_SERVEUR  # clé stockée dans les secrets, jamais affichée
        st.success("Clé API configurée ✅")
    else:
        cle = st.text_input(
            "Clé API Groq", type="password",
            help="Gratuite, sans carte bancaire : https://console.groq.com/keys",
        )
    langue = st.selectbox("Langue de la réunion", ["fr", "en", "auto"], index=0)
    modele_stt = st.selectbox(
        "Modèle de transcription", ["whisper-large-v3", "whisper-large-v3-turbo"], index=0,
        help="large-v3 = plus précis ; turbo = plus rapide.",
    )
    modele_llm = st.selectbox(
        "Modèle pour le résumé", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"], index=0,
        help="120b = meilleur résumé ; 20b = plus rapide.",
    )
    vocab = st.text_area(
        "Vocabulaire technique (optionnel)",
        placeholder="Noms, acronymes, outils… ex : ETL, Looker, BigQuery, facturation électronique",
        help="Aide Whisper à bien écrire les termes propres à ton équipe.",
    )
    st.markdown(
        "---\n**Palier gratuit Groq** : ~2 h d'audio par heure, ~8 h par jour. "
        "Au-delà, l'app attend et relance toute seule."
    )

fichier = st.file_uploader(
    "Fichier audio de la réunion",
    type=["mp3", "m4a", "wav", "ogg", "opus", "webm", "mp4", "flac", "aac", "wma", "amr", "3gp"],
)

if fichier and st.button("🚀 Transcrire et résumer", type="primary", disabled=not cle):
    statut = st.empty()
    barre = st.progress(0.0, text="Préparation de l'audio…")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "source" + Path(fichier.name).suffix)
            with open(src, "wb") as f:
                f.write(fichier.getbuffer())

            total = duree_audio(src)
            morceaux = decouper(src, tmp)
            n = len(morceaux)
            statut.info(f"Durée : {hms(total)} — {n} morceau(x) de 10 min à transcrire.")

            # 1) Transcription
            lignes, textes = [], []
            for i, m in enumerate(morceaux):
                barre.progress(i / (n * 2), text=f"Transcription {i + 1}/{n}…")
                segs = transcrire(m, cle, modele_stt, None if langue == "auto" else langue, vocab, statut)
                decalage = i * CHUNK_SEC
                lignes += [f"[{hms(s['start'] + decalage)}] {s['text']}" for s in segs if s["text"]]
                textes.append(" ".join(s["text"] for s in segs))

            # 2) Notes par partie, puis compte rendu final
            notes = []
            for i, t in enumerate(textes):
                barre.progress((n + i) / (n * 2), text=f"Analyse de la partie {i + 1}/{n}…")
                if t.strip():
                    deb, fin = hms(i * CHUNK_SEC), hms(min((i + 1) * CHUNK_SEC, total))
                    notes.append(f"### Partie {i + 1} ({deb} → {fin})\n" +
                                 chat(cle, modele_llm, SYS_NOTES, t, statut, max_tokens=2000))

            barre.progress(0.97, text="Rédaction du compte rendu…")
            resume = chat(cle, modele_llm, SYS_FINAL, "\n\n".join(notes), statut, max_tokens=4000)

            st.session_state["resultat"] = {
                "nom": Path(fichier.name).stem,
                "duree": hms(total),
                "transcription": "\n".join(lignes),
                "resume": resume,
            }
        barre.progress(1.0, text="Terminé ✅")
        statut.empty()
    except Exception as e:
        barre.empty()
        statut.error(str(e))

if not cle:
    st.info("Colle ta clé API Groq dans le panneau de gauche pour commencer.")

res = st.session_state.get("resultat")
if res:
    st.subheader(f"Résultat — {res['nom']} ({res['duree']})")
    onglet_resume, onglet_transcription = st.tabs(["📝 Compte rendu", "📄 Transcription complète"])
    with onglet_resume:
        st.markdown(res["resume"])
        st.download_button("Télécharger le compte rendu (.md)", res["resume"],
                           file_name=f"{res['nom']}_compte_rendu.md")
    with onglet_transcription:
        st.text_area("Transcription", res["transcription"], height=500, label_visibility="collapsed")
        st.download_button("Télécharger la transcription (.txt)", res["transcription"],
                           file_name=f"{res['nom']}_transcription.txt")
