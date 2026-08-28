# Origine du projet

Sound-Embeddings n'a pas commencé comme un moteur de recherche. Il a commencé
comme une question posée sur un seul album.

Le notebook d'exploration qui a produit ces résultats est dans un dépôt séparé :
[Currents-Music-Similarity](https://github.com/DimitryPopov/Currents-Music-Similarity).

## La question

Pendant un stage de recherche au laboratoire ERTIM (INALCO), je travaillais sur
wav2vec 2.0 et les représentations auto-supervisées de la parole. Le modèle est
entraîné exclusivement sur de la voix, sans transcription, à prédire des
morceaux de signal masqués.

D'où la question : que « voit » un tel modèle quand on lui donne de la musique ?
Il n'a jamais entendu de batterie ni de synthétiseur. Il produit pourtant une
représentation. De quoi est-elle faite ?

## Le protocole

Terrain d'essai : *Currents* de Tame Impala, treize morceaux, 154 230 trames de
20 ms.

1. Extraire les représentations de chaque trame à une couche donnée du modèle.
2. Les discrétiser par k-means en 50 *pseudo-tokens* : chaque trame reçoit le
   numéro du centroïde le plus proche.
3. Caractériser chaque morceau par sa distribution sur ces 50 tokens.
4. Comparer les morceaux par similarité cosinus entre leurs distributions.

Les étapes 1 à 3 transposent à la musique un protocole standard du traitement de
la parole. L'étape 4 reprend l'idée de la métrique ATDS, sur laquelle je
travaillais au laboratoire : deux enregistrements sont proches quand leurs
distributions d'unités acoustiques le sont.

## Le choix de la couche

Le premier essai utilisait `last_hidden_state`, c'est-à-dire la sortie de la
douzième et dernière couche du transformer. Les résultats étaient inertes : la
distribution des tokens était presque uniforme, et rien ne ressortait.

C'est cohérent après coup la couche 12 est celle sur laquelle porte
directement la loss contrastive, donc la plus spécialisée à la structure de la
parole. Les couches basses restent plus proches du signal acoustique lui-même.

Le passage à la couche 2 a tout changé.

## Le moment où ça a marché

La vérification décisive n'était pas une métrique. Elle consistait à récupérer
toutes les trames assignées à un même token, à recoller les fragments d'audio
correspondants, et à écouter le résultat.

Le token 40 fait entendre le synthétiseur de *Gossip*. Le token 34, une famille
de percussions.

Un modèle entraîné uniquement sur de la parole avait construit, sans supervision
et sans avoir jamais entendu de musique, des catégories de timbre identifiables
à l'oreille. C'est le résultat qui a justifié la suite.

## Ce que l'album ne pouvait pas dire

Trois limites sont apparues rapidement.

Les similarités entre les treize morceaux allaient de 0,14 à 0,92, avec une
médiane à 0,77. Autrement dit, tout ressemblait à tout attendu pour un album
d'un même artiste, mais inexploitable pour juger la méthode.

Surtout, il n'existait aucune référence externe. La heatmap de similarité était
une belle figure sans point de comparaison : rien ne permettait de dire si elle
avait raison.

Un test sur un morceau extérieur — *Endors Toi*, de l'album *Lonerism* — a
montré que le vocabulaire appris sur *Currents* transférait à de l'audio jamais
vu, avec des voisins à 0,88. Encourageant, mais toujours invérifiable.

## Le passage à l'échelle

D'où Sound-Embeddings : le même protocole appliqué à 8 000 morceaux de FMA
répartis en huit genres. Les étiquettes de genre fournissent enfin la référence
externe qui manquait, et permettent de remplacer « cette figure a l'air juste »
par une précision@5 mesurée contre le hasard.

Trois choses ont changé au passage :

- le vocabulaire devient un **artefact figé**, appris une fois et appliqué
  partout, sans quoi un token n'a pas le même sens d'une exécution à l'autre ;
- les tokens sont stockés en base plutôt qu'en mémoire, ce qui rend les
  analyses interrogeables en SQL ;
- chaque exécution est identifiée, ce qui permet de comparer plusieurs
  configurations dans la même base c'est ainsi que K = 50, 256 et 1024 ont pu
  être mesurés côte à côte.

Le résultat de cette montée en charge est dans le [README](../README.md).

