# Sound-Embeddings

Moteur de recherche par similarité musicale construit sur les représentations
internes d'un modèle de parole. Aucune étiquette n'est utilisée à l'apprentissage.

## Résultat

Sur 8 000 morceaux du corpus FMA répartis en huit genres, le moteur retrouve un
morceau du même genre dans **40 % des cas** parmi les cinq plus proches voisins,
contre 12,5 % pour un tirage au hasard soit 3,2 fois mieux, sans qu'aucun
genre n'ait été montré au système.

| Genre | Précision@5 |
|---|---|
| Folk | 0,635 |
| International | 0,473 |
| Electronic | 0,473 |
| Rock | 0,462 |
| Hip-Hop | 0,409 |
| Instrumental | 0,393 |
| Pop | 0,244 |
| Experimental | 0,136 |
| **Tous genres** | **0,400** |
| *hasard* | *0,125* |

La précision@5 est la proportion, parmi les cinq voisins renvoyés, de morceaux
partageant le genre de la requête. Mesurée sur 189 requêtes, vocabulaire de
1024 tokens avec pondération TF-IDF.

L'écart entre Folk et Experimental n'est pas un artefact : le modèle est
entraîné sur de la parole, et il excelle sur les genres vocaux et acoustiques
tout en échouant sur une catégorie fourre-tout sans signature acoustique commune.

## Origine

Ce projet croise deux choses apprises séparément.

Pendant un stage de recherche au laboratoire ERTIM (INALCO), j'ai travaillé sur
wav2vec 2.0 et les représentations auto-supervisées de la parole. En cours de
bases de données, j'ai appris SQL. Sound-Embeddings est né de la question de
savoir ce que donnerait le croisement des deux : discrétiser les représentations
d'un modèle de parole en « pseudo-tokens », puis traiter la recherche musicale
comme une recherche documentaire classique, en SQL.

Le projet a commencé sur un seul album, et le moment où il a basculé tient à une
écoute : certains tokens font entendre un timbre de synthétiseur identifiable,
d'autres une famille percussive. Ce récit est dans
[docs/origine.md](docs/origine.md).

Le fonctionnement de wav2vec 2.0 et le lien théorique entre son quantizer et la
méthode employée ici sont détaillés dans [docs/wav2vec2.pdf](docs/wav2vec2.pdf).

Le notebook d'origine, sur l'album *Currents*, est dans
[Currents-Music-Similarity](https://github.com/DimitryPopov/Currents-Music-Similarity).
## Fonctionnement

```
audio (mp3, wav)
  └─ wav2vec 2.0 gelé, couche 2   → (n_trames, 768), une trame = 20 ms
       └─ k-means (vocabulaire)   → un token par trame
            └─ DuckDB             → frames(vocabulary_id, track_id, frame_idx, token_id)
                 └─ SQL           → profils, similarité cosinus, recherche
```

**Les vecteurs ne sont jamais stockés.** Un morceau de quatre minutes produit
12 000 vecteurs de dimension 768 ; sur un gros corpus cela ferait des
téraoctets. Les tokens, eux, tiennent dans un `USMALLINT` : les 12 millions de
trames du corpus occupent quelques dizaines de mégaoctets. Les vecteurs sont un
intermédiaire de calcul, pas une donnée.

**Le vocabulaire est un artefact figé.** Il est appris une fois sur un
échantillon du corpus, sauvegardé en `.npy` avec son manifeste, puis appliqué
tel quel à tout nouvel audio. Sans cela, un token ne signifierait rien d'une
exécution à l'autre. Chaque trame porte le `vocabulary_id` qui lui donne son
sens, ce qui permet à plusieurs vocabulaires de cohabiter dans la même base et
d'être comparés.

## Utilisation

```bash
pip install torch torchaudio transformers scikit-learn duckdb pandas tqdm

python3 learn_vocabulary.py        # apprend le vocabulaire  (~2 min)
python3 index.py                   # indexe le corpus        (~30 min, 8000 morceaux)
python3 search.py "titre"          # voisins d'un morceau indexé
python3 search.py mon_fichier.mp3  # voisins d'un audio hors catalogue
python3 search.py --eval           # mesure la précision@5
```

Une recherche renvoie les voisins avec leur genre et leur similarité :

```
$ python3 search.py data/fma_small/000/000002.mp3

Voisins du fichier : 000002.mp3

  Food                                   Hip-Hop        1.0000
  Motivation (feat. Carolina Deslandes)  Hip-Hop        0.7509
  Fortitude                              Hip-Hop        0.7258
  From The Heart (feat. Richie Campbell) Hip-Hop        0.7200
  Richter Scale                          Hip-Hop        0.6895
```

Le morceau interrogé étant lui-même indexé, il ressort en tête à 1,0000 — les
quatre suivants sont les vrais voisins.

Le corpus FMA se télécharge depuis [mdeff/fma](https://github.com/mdeff/fma).
Aucun fichier audio n'est distribué avec ce dépôt.

## Expériences

Chaque configuration a été mesurée avec la même métrique, sur le même
échantillon de requêtes.

| Vocabulaire | Sans pondération | Avec TF-IDF |
|---|---|---|
| K = 50 | 0,3132 | 0,3143 |
| K = 256 | 0,3619 | 0,3725 |
| K = 1024 | 0,3905 | **0,4000** |

**La granularité du vocabulaire est le levier principal.** Passer de 50 à 1024
tokens fait gagner près de 8 points, avec des rendements décroissants le
plateau n'est pas encore atteint.

**Le TF-IDF n'a pas tenu ses promesses.** L'hypothèse venait de la recherche
textuelle : pondérer chaque token par son IDF (Spärck Jones, 1972) pour atténuer
les unités présentes partout, comme on écarte les mots vides. À K = 50 l'effet
est nul. L'analogie a une limite car un token acoustique fréquent décrit malgré
tout du son, et le pénaliser retire autant de signal que de bruit. Le gain
devient perceptible mais reste faible aux grands vocabulaires, quand les unités
sont assez spécifiques pour qu'une pondération ait quelque chose à distinguer.

**La couche 2 plutôt que la couche 12.** La couche haute est celle sur laquelle
porte la loss contrastive, donc la plus spécialisée à la parole. En recollant
l'audio des trames assignées à un même token, on entend des timbres
identifiables : un synthétiseur, une famille percussive et non des unités
sub-phonémiques.

## Limites

- **Sac de tokens.** Un morceau est représenté par la distribution de ses
  tokens, sans leur ordre : joué à l'envers, il aurait le même vecteur. Passer
  aux bigrammes capturerait un peu de structure temporelle.
- **Fenêtrage sans recouvrement.** L'inférence se fait par blocs de 20 s ; les
  trames de bord voient un contexte tronqué.
- **Un seul genre par morceau.** L'évaluation repose sur le `genre_top` de FMA,
  qui simplifie la réalité musicale.
- **Modèle entraîné sur de la parole.** Le transfert à la musique est l'objet de
  l'expérience, pas une hypothèse validée. Un modèle entraîné sur de l'audio
  général donnerait probablement de meilleurs résultats — au prix de la
  compréhension fine que j'ai du modèle employé ici.

## Structure

```
learn_vocabulary.py   extraction wav2vec 2.0, k-means, sauvegarde du vocabulaire
index.py              ingestion du corpus, assignation des tokens, écriture DuckDB
search.py             recherche par titre ou par fichier, évaluation
schema.sql            tables, vues de similarité et de pondération
vocabulary.json       manifeste du vocabulaire (modèle, couche, K, graine)
docs/                 carnet de recherche (PDF) et origine du projet
```