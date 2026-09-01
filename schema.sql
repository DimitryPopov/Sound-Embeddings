-- Sound-Embeddings: similarity search index.
-- One frame = 20 ms of audio. Only tokens are stored, never the 768-dim
-- vectors, which stay an intermediate result.

-- DuckDB has no auto-increment: sequences fill that role.
CREATE SEQUENCE IF NOT EXISTS vocabulary_id_seq;
CREATE SEQUENCE IF NOT EXISTS track_id_seq;

-- The vocabulary: one .npy file plus the parameters that produced it.
CREATE TABLE IF NOT EXISTS vocabularies (
    vocabulary_id BIGINT PRIMARY KEY DEFAULT nextval('vocabulary_id_seq'),
    path          VARCHAR NOT NULL,        -- path to vocabulary.npy
    model         VARCHAR NOT NULL,
    layer         INTEGER NOT NULL,
    k             INTEGER NOT NULL,
    seed          INTEGER NOT NULL,
    created_at    TIMESTAMP DEFAULT current_timestamp
);

-- The catalogue.
CREATE TABLE IF NOT EXISTS tracks (
    track_id    BIGINT PRIMARY KEY DEFAULT nextval('track_id_seq'),
    source      VARCHAR NOT NULL,          -- 'fma_small', 'local', 'jamendo'...
    external_id VARCHAR NOT NULL,
    title       VARCHAR,
    artist      VARCHAR,
    genre       VARCHAR,                   -- genre_top pour FMA
    path        VARCHAR NOT NULL,
    sha256      VARCHAR,
    n_frames    INTEGER,
    indexed_at  TIMESTAMP DEFAULT current_timestamp,
    UNIQUE (source, external_id)           -- makes indexing replayable
);

-- The raw fact: one token per frame.
CREATE TABLE IF NOT EXISTS frames (
    vocabulary_id BIGINT    NOT NULL,
    track_id      BIGINT    NOT NULL,
    frame_idx     INTEGER   NOT NULL,       -- LOCAL to the track, from 0
    token_id      USMALLINT NOT NULL        -- K < 65536
);
-- Track profile: share of each token among its frames.
CREATE OR REPLACE VIEW profiles AS
SELECT
    vocabulary_id,
    track_id,
    token_id,
    COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER (PARTITION BY vocabulary_id, track_id) AS p
FROM frames
GROUP BY vocabulary_id, track_id, token_id;

CREATE OR REPLACE VIEW profile_norms AS
SELECT vocabulary_id, track_id, sqrt(SUM(p * p)) AS norm
FROM profiles
GROUP BY vocabulary_id, track_id;

-- Cosine between profiles. Self-join on token_id: tokens absent from a track
-- have p = 0 and contribute nothing to the dot product.
-- ATTENTION : cette vue est un produit croise. Sur 8000 morceaux elle represente
-- 64 millions de paires : ne l'interrogez JAMAIS sans filtrer sur track_a.
CREATE OR REPLACE VIEW similarity AS
SELECT
    a.vocabulary_id,
    a.track_id AS track_a,
    b.track_id AS track_b,
    SUM(a.p * b.p) / (na.norm * nb.norm) AS cosine
FROM profiles a
JOIN profiles b
  ON a.token_id = b.token_id AND a.vocabulary_id = b.vocabulary_id
JOIN profile_norms na ON na.vocabulary_id = a.vocabulary_id AND na.track_id = a.track_id
JOIN profile_norms nb ON nb.vocabulary_id = b.vocabulary_id AND nb.track_id = b.track_id
GROUP BY a.vocabulary_id, a.track_id, b.track_id, na.norm, nb.norm;

-- Statistiques par token sur l'ensemble du corpus.
CREATE OR REPLACE VIEW token_stats AS
SELECT vocabulary_id, token_id,
       count(DISTINCT track_id) AS df,      -- nb de morceaux contenant le token
       avg(p)                   AS mean_p   -- part moyenne du token
FROM profiles
GROUP BY vocabulary_id, token_id;

-- Poids de chaque token. Choisir UNE des deux definitions de weight.
CREATE OR REPLACE VIEW token_weights AS
SELECT s.vocabulary_id,
       s.token_id,
        -- ln(1.0 / s.mean_p) AS weight            -- frequence de collection inverse
       ln(n.total::DOUBLE / s.df) AS weight       -- IDF classique (Sparck Jones, 1972) (explication dans le pdf extrait carnet de recherche)
FROM token_stats s
JOIN (SELECT vocabulary_id, count(DISTINCT track_id) AS total
      FROM profiles GROUP BY vocabulary_id) n
  ON n.vocabulary_id = s.vocabulary_id;

-- Profils ponderes : meme structure que 'profiles', d'ou la reutilisation
-- possible de toutes les requetes existantes.
CREATE OR REPLACE VIEW profiles_tfidf AS
SELECT p.vocabulary_id, p.track_id, p.token_id, p.p * w.weight AS p
FROM profiles p
JOIN token_weights w
  ON w.vocabulary_id = p.vocabulary_id AND w.token_id = p.token_id;

CREATE OR REPLACE VIEW profile_norms_tfidf AS
SELECT vocabulary_id, track_id, sqrt(sum(p * p)) AS norm
FROM profiles_tfidf
GROUP BY vocabulary_id, track_id;
