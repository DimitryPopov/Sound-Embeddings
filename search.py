"""Search the index: by track title, by an audio file, or evaluate the engine."""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from learn_vocabulary import extraire_features, assigner, K

ROOT      = Path(__file__).resolve().parent
DB_PATH   = str(ROOT / "index.duckdb")
VOCAB_NPY = str(ROOT / "vocabulary.npy")


def latest_vocabulary(con):
    return con.execute("SELECT max(vocabulary_id) FROM vocabularies").fetchone()[0]


def find_track(con, query):
    rows = con.execute(
        "SELECT track_id, title FROM tracks WHERE title ILIKE ? ORDER BY title LIMIT 20",
        ["%" + query + "%"],
    ).fetchall()
    if not rows:
        raise SystemExit("Aucun morceau ne correspond a : %s" % query)
    if len(rows) > 1:
        print("Plusieurs correspondances, la premiere est retenue :")
        for _, title in rows[:5]:
            print("   ", title)
    return rows[0]


def neighbours(con, vocabulary_id, track_id, limit=5):
    return con.execute(
        """
        SELECT tb.title, tb.genre, round(s.cosine, 4) AS cosine
        FROM similarity s
        JOIN tracks tb ON tb.track_id = s.track_b
        WHERE s.vocabulary_id = ? AND s.track_a = ? AND s.track_b <> s.track_a
        ORDER BY s.cosine DESC
        LIMIT ?
        """,
        [vocabulary_id, track_id, limit],
    ).fetchall()


def search_by_file(con, vocabulary_id, path, limit=5):
    """Meme calcul que la vue 'similarity', mais un cote vient d'un fichier."""
    C = np.load(VOCAB_NPY)
    tokens = assigner(extraire_features(path), C)
    profile = np.bincount(tokens, minlength=K) / len(tokens)

    con.register("query_profile",
                 pd.DataFrame({"token_id": np.arange(K), "p": profile}))
    rows = con.execute(
        """
        SELECT t.title, t.genre,
               round(SUM(q.p * b.p)
                     / ((SELECT sqrt(SUM(p * p)) FROM query_profile) * nb.norm), 4)
                   AS cosine
        FROM query_profile q
        JOIN profiles b
          ON b.token_id = q.token_id AND b.vocabulary_id = ?
        JOIN profile_norms nb
          ON nb.vocabulary_id = b.vocabulary_id AND nb.track_id = b.track_id
        JOIN tracks t ON t.track_id = b.track_id
        GROUP BY t.title, t.genre, nb.norm
        ORDER BY cosine DESC
        LIMIT ?
        """,
        [vocabulary_id, limit],
    ).fetchall()
    con.unregister("query_profile")
    return rows


def evaluer(con, vocabulary_id, k=5, modulo=40, tfidf=False):
    """Precision@k sur le genre : la seule mesure qui juge le moteur.

    Pour chaque morceau d'un echantillon, on regarde les k voisins les plus
    proches et on compte ceux qui partagent son genre. Le hasard donnerait
    1 / nombre de genres.
    """
    vue_p = "profiles_tfidf" if tfidf else "profiles"
    vue_n = "profile_norms_tfidf" if tfidf else "profile_norms"

    sql = """
    WITH ech AS (
        SELECT track_id FROM tracks
        WHERE genre IS NOT NULL AND hash(track_id) %% %d = 0
    ),
    pa AS (
        SELECT pr.track_id, pr.token_id, pr.p
        FROM %s pr JOIN ech e ON e.track_id = pr.track_id
        WHERE pr.vocabulary_id = ?
    ),
    pb AS (
        SELECT track_id, token_id, p FROM %s WHERE vocabulary_id = ?
    ),
    sim AS (
        SELECT a.track_id AS ta, b.track_id AS tb,
               SUM(a.p * b.p) / (na.norm * nb.norm) AS cosine
        FROM pa a
        JOIN pb b ON a.token_id = b.token_id AND a.track_id <> b.track_id
        JOIN %s na ON na.track_id = a.track_id AND na.vocabulary_id = ?
        JOIN %s nb ON nb.track_id = b.track_id AND nb.vocabulary_id = ?
        GROUP BY a.track_id, b.track_id, na.norm, nb.norm
    ),
    top AS (
        SELECT ta, tb, cosine,
               row_number() OVER (PARTITION BY ta ORDER BY cosine DESC) AS rang
        FROM sim QUALIFY rang <= %d
    )
    SELECT ga.genre,
           count(DISTINCT top.ta) AS n_requetes,
           round(avg(CASE WHEN ga.genre = gb.genre THEN 1.0 ELSE 0.0 END), 4) AS precision_k
    FROM top
    JOIN tracks ga ON ga.track_id = top.ta
    JOIN tracks gb ON gb.track_id = top.tb
    GROUP BY ROLLUP (ga.genre)
    ORDER BY ga.genre NULLS LAST
    """ % (modulo, vue_p, vue_p, vue_n, vue_n, k)

    n_genres = con.execute(
        "SELECT count(DISTINCT genre) FROM tracks WHERE genre IS NOT NULL"
    ).fetchone()[0]

    rows = con.execute(sql, [vocabulary_id] * 4).fetchall()
    print("\n-- Precision@%d par genre %s ------------------------------"
          % (k, "(TF-IDF)" if tfidf else "(brut)"))
    for genre, n, precision in rows:
        libelle = genre if genre is not None else "TOUS GENRES"
        print("  %-22s %4d requetes   %.4f" % (libelle, n, precision))
    print("\n  hasard (1/%d genres) : %.4f" % (n_genres, 1.0 / n_genres))

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "--eval"

    if arg == "--eval":
        con = duckdb.connect(DB_PATH, read_only=True)
        vid = latest_vocabulary(con)
        evaluer(con, vid, tfidf=False)
        evaluer(con, vid, tfidf=True)
        con.close()
        return

    as_file = Path(arg).exists()
    # register() interdit en lecture seule : mode ecriture pour la recherche fichier
    con = duckdb.connect(DB_PATH, read_only=not as_file)
    vocabulary_id = latest_vocabulary(con)

    if as_file:
        print("\nVoisins du fichier : %s\n" % Path(arg).name)
        rows = search_by_file(con, vocabulary_id, Path(arg))
    else:
        track_id, title = find_track(con, arg)
        print("\nVoisins de : %s\n" % title)
        rows = neighbours(con, vocabulary_id, track_id)

    for title, genre, cosine in rows:
        print("  %-38s %-14s %.4f" % (str(title)[:38], str(genre)[:14], cosine))
    con.close()


if __name__ == "__main__":
    main()