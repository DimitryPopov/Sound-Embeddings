"""Learn the acoustic vocabulary once, save it as a reusable artifact."""

import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# --- Configuration ----------------------------------------------------------
ROOT   = Path(__file__).resolve().parent
device = "mps" if torch.backends.mps.is_available() else "cpu"
SR     = 16000
WIN    = 20 * SR          # taille de fenetre en echantillons
K      = 1024               # choix empirique
LAYER  = 2                # porte l'information musicale
SEED   = 0
budget = 200_000          # trames max retenues par morceau

VOCAB_NPY  = ROOT / "vocabulary.npy"
VOCAB_JSON = ROOT / "vocabulary.json"

FMA_AUDIO  = ROOT / "data" / "fma_small"
N_TRAINING = 400          # morceaux tires au hasard pour apprendre le vocabulaire

# --- Modele -----------------------------------------------------------------
fe    = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model = model.to(device)
model = model.eval()


# --- Fonctions reutilisables ------------------------------------------------
def extraire_features(path):
    """Un morceau -> (n_trames, 768) a la couche LAYER."""
    sig, sr = torchaudio.load(path)
    sig = sig.mean(0, keepdim=True)
    if sr != SR:
        sig = torchaudio.transforms.Resample(sr, SR)(sig)
    sig = sig.squeeze(0)

    feats = []
    for k in range(0, sig.shape[0], WIN):
        chunk  = sig[k:k + WIN]
        inputs = fe(chunk.numpy(), sampling_rate=SR,
                    return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            h = model(inputs, output_hidden_states=True).hidden_states[LAYER]
        feats.append(h.squeeze(0).cpu())

    return torch.cat(feats, dim=0).numpy()


def assigner(H, C, bloc=8192):
    """Chaque trame -> le token du centroide le plus proche."""
    c2 = (C ** 2).sum(axis=1)
    out = np.empty(len(H), dtype=np.uint16)
    for i in range(0, len(H), bloc):
        X = H[i:i + bloc]
        out[i:i + bloc] = (c2 - 2 * X @ C.T).argmin(axis=1)
    return out


# --- Apprentissage : ne tourne QUE si on lance ce fichier directement --------
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    tous_les_mp3 = sorted(FMA_AUDIO.rglob("*.mp3"))
    if len(tous_les_mp3) < N_TRAINING:
        raise SystemExit("Seulement %d mp3 trouves dans %s"
                         % (len(tous_les_mp3), FMA_AUDIO))
    tirage = rng.choice(len(tous_les_mp3), N_TRAINING, replace=False)
    paths_apprentissage = [tous_les_mp3[i] for i in tirage]
    print("%d mp3 dans le corpus, %d retenus pour l'apprentissage"
          % (len(tous_les_mp3), N_TRAINING))

    km = MiniBatchKMeans(n_clusters=K, random_state=SEED, batch_size=4096)
    utilises, ignores, total_trames = [], 0, 0
    H_dernier = None          # garde pour la verification, evite une inference

    boucle = tqdm(paths_apprentissage, desc="apprentissage")
    for p in boucle:
        try:
            H = extraire_features(p)
        except Exception as exc:          # mp3 corrompus signales par le depot
            tqdm.write("  ignore %s : %s" % (p.name, exc))
            ignores += 1
            continue

        if len(H) < K:
            ignores += 1
            continue

        H_dernier = H                     # avant sous-echantillonnage
        if len(H) > budget:
            H = H[rng.choice(len(H), budget, replace=False)]

        km.partial_fit(H)
        utilises.append(p)
        total_trames += len(H)
        boucle.set_postfix(trames=total_trames, ignores=ignores)

    if not utilises:
        raise SystemExit("Aucun morceau exploitable.")

    C = km.cluster_centers_.astype(np.float32)
    print("\nvocabulaire :", C.shape)
    print("morceaux utilises : %d, ignores : %d, trames vues : %d"
          % (len(utilises), ignores, total_trames))

    np.save(VOCAB_NPY, C)

    manifest = {
        "model": "facebook/wav2vec2-base",
        "layer": LAYER,
        "k": K,
        "seed": SEED,
        "dim": int(C.shape[1]),
        "frames_budget": budget,
        "training_corpus": "fma_small",
        "training_tracks_count": len(utilises),
        "training_frames": total_trames,
        "training_tracks": [p.name for p in utilises],
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    VOCAB_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("accord avec km.predict :",
          (assigner(H_dernier, C) == km.predict(H_dernier)).mean())

    C2 = np.load(VOCAB_NPY)
    print("stable apres rechargement :",
          (assigner(H_dernier, C2) == assigner(H_dernier, C)).all())