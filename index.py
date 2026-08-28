"""Index the catalogue: audio -> tokens -> DuckDB."""

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from learn_vocabulary import extraire_features, assigner

ROOT       = Path(__file__).resolve().parent
DB_PATH    = str(ROOT / "index.duckdb")
VOCAB_NPY  = "vocabulary.npy"
VOCAB_JSON = ROOT / "vocabulary.json"

FMA_AUDIO    = ROOT / "data" / "fma_small"
FMA_METADATA = ROOT / "data" / "fma_metadata" / "tracks.csv"
SOURCE       = "fma_small"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def charger_metadonnees():
    """tracks.csv -> {track_id: (titre, artiste, genre)}. En-tete a deux niveaux."""
    df = pd.read_csv(FMA_METADATA, index_col=0, header=[0, 1])
    meta = {}
    for track_id, row in df.iterrows():
        meta[int(track_id)] = (
            row.get(("track", "title")),
            row.get(("artist", "name")),
            row.get(("track", "genre_top")),
        )
    return meta


def chemin_audio(track_id):
    tid = "%06d" % track_id
    return FMA_AUDIO / tid[:3] / (tid + ".mp3")


def get_or_create_vocabulary(con, manifest):
    """Cherche le vocabulaire, l'insere s'il est absent. Renvoie son id."""
    row = con.execute(
        "SELECT vocabulary_id FROM vocabularies "
        "WHERE path = ? AND layer = ? AND k = ? AND seed = ?",
        [VOCAB_NPY, manifest["layer"], manifest["k"], manifest["seed"]],
    ).fetchone()
    if row:
        return row[0]
    return con.execute(
        "INSERT INTO vocabularies (path, model, layer, k, seed) "
        "VALUES (?, ?, ?, ?, ?) RETURNING vocabulary_id",
        [VOCAB_NPY, manifest["model"], manifest["layer"],
         manifest["k"], manifest["seed"]],
    ).fetchone()[0]


def deja_indexe(con, vocabulary_id, external_id):
    row = con.execute(
        "SELECT t.track_id FROM tracks t "
        "WHERE t.source = ? AND t.external_id = ? "
        "  AND EXISTS (SELECT 1 FROM frames f "
        "              WHERE f.track_id = t.track_id AND f.vocabulary_id = ?)",
        [SOURCE, external_id, vocabulary_id],
    ).fetchone()
    return row is not None


def index_track(con, vocabulary_id, C, path, titre, artiste, genre):
    """Extrait, assigne et ecrit les trames d'un morceau. Renvoie le nb de trames."""
    external_id = path.stem

    H = extraire_features(path)
    tokens = assigner(H, C)

    row = con.execute(
        "SELECT track_id FROM tracks WHERE source = ? AND external_id = ?",
        [SOURCE, external_id],
    ).fetchone()

    if row:
        track_id = row[0]
        con.execute("UPDATE tracks SET n_frames = ? WHERE track_id = ?",
                    [len(tokens), track_id])
    else:
        track_id = con.execute(
            "INSERT INTO tracks "
            "(source, external_id, title, artist, genre, path, sha256, n_frames) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING track_id",
            [SOURCE, external_id, titre, artiste, genre,
             str(path), sha256(path), len(tokens)],
        ).fetchone()[0]

    frames_df = pd.DataFrame({
        "vocabulary_id": np.full(len(tokens), vocabulary_id, dtype=np.int64),
        "track_id":      np.full(len(tokens), track_id, dtype=np.int64),
        "frame_idx":     np.arange(len(tokens), dtype=np.int32),
        "token_id":      tokens.astype(np.uint16),
    })
    con.register("frames_df", frames_df)
    con.execute("INSERT INTO frames "
                "SELECT vocabulary_id, track_id, frame_idx, token_id FROM frames_df")
    con.unregister("frames_df")

    return len(tokens)


def main():
    con = duckdb.connect(DB_PATH)
    con.execute((ROOT / "schema.sql").read_text())

    manifest = json.loads(VOCAB_JSON.read_text())
    vocabulary_id = get_or_create_vocabulary(con, manifest)
    C = np.load(ROOT / VOCAB_NPY)

    meta = charger_metadonnees()
    mp3 = sorted(FMA_AUDIO.rglob("*.mp3"))
    print("%d mp3 trouves, %d entrees de metadonnees" % (len(mp3), len(meta)))

    faits, sautes, echecs, total_trames = 0, 0, 0, 0
    boucle = tqdm(mp3, desc="indexation")

    for path in boucle:
        external_id = path.stem
        if deja_indexe(con, vocabulary_id, external_id):
            sautes += 1
            continue

        titre, artiste, genre = meta.get(int(external_id), (None, None, None))
        try:
            total_trames += index_track(con, vocabulary_id, C, path,
                                        titre, artiste, genre)
            faits += 1
        except Exception as exc:
            tqdm.write("  echec %s : %s" % (path.name, exc))
            echecs += 1

        boucle.set_postfix(faits=faits, sautes=sautes, echecs=echecs)

    print("\nindexes : %d, sautes : %d, echecs : %d" % (faits, sautes, echecs))

    print("\n-- Verification -------------------------------------------")
    ecarts = con.execute(
        "SELECT count(*) FROM ("
        "  SELECT t.track_id FROM tracks t JOIN frames f ON f.track_id = t.track_id"
        "  WHERE f.vocabulary_id = ?"
        "  GROUP BY t.track_id, t.n_frames HAVING count(*) <> t.n_frames)",
        [vocabulary_id],
    ).fetchone()[0]
    print("morceaux dont n_frames ne correspond pas :", ecarts)

    print("\n-- Genres -------------------------------------------------")
    for genre, n in con.execute(
        "SELECT coalesce(genre, '(inconnu)'), count(*) FROM tracks "
        "WHERE source = ? GROUP BY 1 ORDER BY 2 DESC", [SOURCE],
    ).fetchall():
        print("  %-22s %5d" % (genre, n))

    total = con.execute(
        "SELECT count(*) FROM frames WHERE vocabulary_id = ?", [vocabulary_id]
    ).fetchone()[0]
    print("\ntotal trames :", total)
    con.close()


if __name__ == "__main__":
    main()