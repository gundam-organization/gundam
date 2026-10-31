# Optimisation des dials standards CUDA : résultats V100 et suite pour Minuit2

31 octobre 2026 — OA2024, `configOA2024.yaml` et `override/v13Prod.yaml`,
Tesla V100-SXM2-32GB, quatre threads hôte. Les calculs restent en `double`.

## Résultat principal

Le compactage des descripteurs accélère le calcul global des réponses des dials
par **2,50**, et divise approximativement par deux le temps d'une évaluation
complète. Il améliore aussi les coordonnées locales coûteuses, notamment les
coordonnées propres FSI. La médiane des variations locales reste proche de
2,09 ms : leur coût dominant est désormais le recalcul des poids événementiels.

Pour les propositions globales d'un MCMC, le gain est directement pertinent.
Il ne mesure cependant pas une accélération d'un MCMC complet. Pour Minuit2,
la priorité suivante devrait être de rendre le recalcul des événements sélectif,
après avoir mesuré la fraction réellement affectée par chaque coordonnée.

## Jeux de mesures et méthode

Les résultats proviennent de deux exécutions complémentaires :

- [Benchmark sans instrumentation](../../benchmark-weights-compact-20261031-190025/report.md) :
  CPU et quatre variantes CUDA, trois essais, après échauffement. Les temps
  d'évaluation sont les médianes de ce lancement.
- [Capture Nsight Compute](../../benchmark-weights-compact-20261031-195643/weights.ncu-rep) :
  NCU 2025.2.1, référence puis compact, chacun sur un appel local et un appel
  global. Les [métriques extraites](../../benchmark-weights-compact-20261031-195643/analysis/metrics.json)
  conservent les compteurs utilisés dans ce rapport.

Les deux exécutions utilisent le même hash du module GUNDAM. Chaque lancement
charge les inputs une seule fois, puis active les backends à chaud. Le benchmark
couvre 842 coordonnées libres en `local1`. Les points et graines sont identiques
entre variantes ; la sélection des données et la vraisemblance sont conservées.
Il s'agit de workloads reproductibles de propagation, sans minimisation ni Hesse.

Les configurations sont exécutées successivement, sans alternance aléatoire.
Les temps absolus de jobs différents ne doivent pas être comparés directement.

## Ce qui a changé dans le backend

Le modèle contient 2 406 926 événements et 20 340 421 dials uniques. Les dials
standards utilisaient sur le GPU un descripteur générique de 176 octets, contenant
également des champs destinés aux programmes, composites et poids externes.

L'option CUDA `compactDials: true` introduit quatre colonnes de `uint32` :

1. référence de l'entrée du dial ;
2. offset du payload ;
3. taille du payload ;
4. référence d'une politique de réponse partagée.

Une politique regroupe le type, le nombre d'entrées, l'extrapolation, les bornes
et le smoothing. OA2024 n'en nécessite que **12**. Leur regroupement utilise les
valeurs binaires, notamment pour préserver les zéros signés.

| Stockage des descripteurs GPU | Référence | Compact |
|---|---:|---:|
| Octets | 3 579 914 096 | 325 448 848 |
| Taille décimale | 3,58 Go | 325 Mo |

La réduction est de **90,9 %**, soit environ **3,25 Go libérés**. Ce tableau
mesure le stockage, pas le trafic mémoire à chaque propagation.

Les payloads, les indices des réponses, les dépendances et l'ordre des produits
par événement sont conservés. Les deux parcours dense et sparse réutilisent les
mêmes routines numériques. Le cache des entrées transformées reste actif.

Le mode compact reste optionnel. Un modèle contenant des formules, composites,
surfaces ou producteurs externes conserve le chemin générique. Un dépassement
des indices 32 bits ou l'absence de gain de stockage impose également ce repli.
Les diagnostics précisent le mode réellement utilisé ; le benchmark refuse un
cas nommé `compact` si cette représentation n'a pas été activée.

Cette étape concerne les descripteurs. Elle ne transpose pas les coefficients
des splines et ne spécialise pas encore les kernels par famille de dial.

## Gains mesurés sans NCU

Temps médians en millisecondes, avec layout `warpInterleaved` :

| Mode et scénario | Référence | Compact | Accélération |
|---|---:|---:|---:|
| Cached, constant | 0,533 | 0,536 | comparable |
| Cached, local1 | 2,089 | 2,091 | comparable |
| Cached, local2 | 2,085 | 2,091 | comparable |
| Cached, global | 14,427 | 7,070 | **2,04×** |
| Recompute, global | 13,961 | 6,818 | **2,05×** |

En cached/global, le détail isole bien l'effet du changement :

| Étage | Référence | Compact |
|---|---:|---:|
| Réponses des dials | 12,251 ms | **4,905 ms** |
| Poids événementiels | 1,255 ms | 1,255 ms |
| Histogrammes | 0,186 ms | 0,187 ms |
| LLH device | 0,011 ms | 0,011 ms |

Le calcul des réponses baisse de **60 %**. Les trois médianes d'évaluation
globale de la référence sont 14,440 / 14,425 / 14,426 ms ; celles du compact sont
7,050 / 7,097 / 7,082 ms. Le gain est largement supérieur à la dispersion
observée. Les très petites différences locales ne permettent pas de conclure
à une régression.

Les colonnes GPU recouvrent le temps d'évaluation complet : elles ne doivent
pas être additionnées au temps hôte.

## Ce qu'apporte NCU

Les timers du backend indiquent **où** le temps est dépensé. NVIDIA Nsight
Compute permet de comprendre **comment** le kernel utilise le GPU : octets
transférés en DRAM, efficacité des secteurs de lecture, caches, registres,
occupation et activité des unités.

Avant modification, les timers seuls ne permettaient pas de départager
interpolation coûteuse, accès mémoire dispersés ou pression sur les registres.
La capture a orienté le travail vers la représentation des données. La nouvelle
capture permet maintenant de vérifier cette explication.

### Réponses denses : moins de trafic, malgré une occupation plus faible

Comparaison des actions 2 et 6 du même rapport NCU :

| Métrique | Référence | Compact |
|---|---:|---:|
| Durée du kernel profilé | 11,662 ms | 4,585 ms |
| Lectures DRAM | **8,182 Go** | **2,439 Go** |
| Écritures DRAM | 326,6 Mo | 326,2 Mo |
| Débit DRAM total | 730 Go/s | 603 Go/s |
| Débit DRAM / pic rapporté | 81,3 % | 67,2 % |
| Octets utiles / secteur de lecture global de 32 octets | 6,73 | 9,15 |
| Taux de hit L2 | 26,8 % | 38,3 % |
| Registres par thread | 52 | 66 |
| Occupation mesurée | 45,4 % | 33,2 % |
| Trafic load/store en mémoire locale | nul | nul |

Les lectures DRAM diminuent de **70,2 %**. Les écritures restent pratiquement
identiques. L'utilisation des secteurs de lecture s'améliore, mais reste
partielle ; ce ratio côté instructions ne constitue pas une efficacité DRAM
globale exacte.

Le compact est plus rapide **alors même que son débit en Go/s et son occupation
sont plus faibles**. Il transfère surtout beaucoup moins de données. Maximiser
isolément ces pourcentages n'aurait donc pas été le bon objectif.

La hausse de 52 à 66 registres limite les blocs résidents par SM de quatre à
trois, pour des blocs de 256 threads. Cela donne une piste supplémentaire pour
les spécialisations futures. Aucun trafic local n'est mesuré : les spills ne
sont pas le coût principal observé.

Les compteurs ne permettent pas d'attribuer chaque octet à un champ précis du
descripteur ou du payload sans analyse supplémentaire des instructions.

### Poids événementiels : le coût restant des appels locaux

Les actions 1, 3, 5 et 7 montrent des résultats voisins : environ **1,22 ms**,
**0,94 Go de lectures DRAM**, **783 à 786 Go/s**, environ **88 % du pic**,
32 registres par thread et 92 % d'occupation.

Le compactage des dials n'a donc pratiquement pas changé cet étage. La capture
locale sélectionne la coordonnée 0 et seulement **50 dials candidats**. Le
kernel des réponses sélectionnées coûte environ 8 µs, mais le kernel suivant
recalcule encore les poids de **tous les événements**.

C'est la distinction importante : une sélection sparse des réponses n'implique
pas encore une sélection sparse des événements.

### Précautions d'interprétation

NCU rejoue les kernels pour collecter les métriques. Ici, le mode est
`kernel`, avec `cache-control all` et `clock-control none` : caches vidés entre
les passes et fréquences non verrouillées. Les durées profilées servent au
diagnostic ; le benchmark sans instrumentation reste la référence pour les
gains applicatifs. Les intervalles CUDA entourant la capture incluent les effets
du profileur et ne constituent pas des temps normaux de propagation.

La capture locale concerne une seule coordonnée, pas toutes les familles de
paramètres ni les coordonnées propres FSI. Voir le
[guide NVIDIA sur les replays et la reproductibilité](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
pour les limites de cette méthode.

## Variations locales : ce qui profite déjà à Minuit2

La médiane locale cache des coordonnées bien plus coûteuses :

| Coordonnée effective du fit | Référence | Compact |
|---|---:|---:|
| FSI `#1_eigen` | 10,898 ms | **6,037 ms** |
| FSI `#0_eigen` | 10,877 ms | **5,983 ms** |
| `NucleonFSI_TotalGraph` | 6,452 ms | **3,592 ms** |
| `MaCCQEGraph` | 4,591 ms | **2,989 ms** |

Une coordonnée propre peut modifier plusieurs paramètres physiques. Le nombre
de coordonnées changées n'est donc pas une mesure suffisante du travail.
Les candidats comptent les dials visités avant réutilisation de leur entrée
transformée, pas nécessairement les interpolations effectivement exécutées.

Minuit2 combine des évaluations locales, notamment pour les différences finies,
et des déplacements suivant des directions qui peuvent modifier plusieurs
coordonnées. Il faudra mesurer leur fréquence dans un fit représentatif avant
de traduire ces gains en accélération du fit complet. Le scénario global est
pertinent pour un MCMC à propositions globales ; d'autres propositions MCMC
peuvent également être locales ou par blocs.

## Prochain chantier recommandé : propagation sélective des événements

Pour privilégier les variations locales, je placerais ce chantier avant une
nouvelle transposition générale des payloads. Les poids événementiels représentent
environ **60 % des 2,09 ms** d'une évaluation locale médiane. Leur kernel approche
déjà la bande passante maximale : éviter les lectures inutiles est une piste
plus directe qu'augmenter encore son occupation.

### 1. Mesurer la couverture événementielle par coordonnée

Ajouter au benchmark résident, hors chronomètre de construction des index :

- nombre de paramètres physiques et de dials sélectionnés ;
- nombre d'événements uniques dépendants et fraction du modèle couverte ;
- taille de l'index inverse nécessaire ;
- temps pour sélectionner les événements et, ensuite, les recalculer.

Un faible nombre de dials peut malgré tout affecter presque tous les événements,
par exemple via des normalisations largement partagées. Les 50 candidats de la
capture ne prouvent donc pas qu'un recalcul partiel serait rentable pour cette
coordonnée. Cette mesure doit précéder le choix de l'index et du seuil dense.

### 2. Recalculer les produits des seuls événements affectés

Construire à l'activation du backend des dépendances inverses entre les
paramètres/dials et les événements. Pour une mise à jour locale, former une
sélection sans doublons, recalculer le produit complet de chaque événement
sélectionné et conserver les autres poids en mémoire GPU.

Il faut préserver l'ordre des multiplications, le poids de base, le cap de
repondération et les cas zéro/négatif. Une mise à jour par ratio
`nouvelle réponse / ancienne réponse` serait fragile en présence de zéros et
modifierait l'historique des arrondis : ce n'est pas la première approche à
retenir.

Une première version peut sélectionner un sur-ensemble conservatif à partir
des dépendances physiques, même si certaines entrées transformées ne changent
finalement pas. Le coût supplémentaire se mesure ; manquer un événement serait
une erreur de résultat.

Le parcours dense doit rester disponible. Il sera choisi quand la fraction
affectée, le coût de sélection ou la taille des index rendent le parcours sparse
moins favorable. Les modèles à migration d'observables, binning dynamique,
producteurs externes ou état invalidé demandent une gestion explicite ; un
repli dense est préférable pour la première version.

### 3. Garder d'abord les histogrammes et la LLH sur leur chemin actuel

Après mise à jour des seuls poids concernés, refaire initialement la réduction
complète des histogrammes à partir du buffer de poids. Cela isole le gain sur
l'étage dominant tout en conservant la réduction actuelle.

Dans une seconde étape, on pourra sélectionner les bins touchés et recalculer
leurs sommes et sommes de carrés, avec l'ordre de réduction existant. Une mise
à jour par deltas atomiques introduirait d'autres questions de déterminisme et
d'accumulation des arrondis. La LLH device ne coûtant qu'environ 11 µs, sa
sélection partielle ne serait pas prioritaire.

### 4. Valider sur des séquences proches de Minuit2

Conserver les cas constant/local1/local2/global et ajouter des séquences
représentatives : points central/plus/moins de différences finies, retours au
centre, alternance des coordonnées, coordonnées propres FSI et déplacements
multidimensionnels. Rejouer ensuite une trace de coordonnées d'un vrai Minuit2,
hors coût de chargement, puis mesurer un fit complet comparable.

Comparer les poids individuels, histogrammes, erreurs et LLH avec le parcours
dense et CPU, y compris après erreur et invalidation. Mesurer aussi le coût de
construction et la mémoire des index : une accélération locale ne doit pas
s'obtenir au prix d'une initialisation ou d'une consommation mémoire excessive.

En supprimant idéalement les seuls 1,25 ms des poids, sans aucun autre coût,
la médiane locale passerait de 2,09 à environ 0,84 ms. C'est une borne de budget,
pas une prévision : la sélection et les événements restant à calculer auront
un coût. Les normalisations globales pourront conserver un coût proche du dense.

## Suite des optimisations globales

Pour les propositions globales et les coordonnées FSI coûteuses, les pistes
restent la transposition des payloads, la spécialisation des `UniformSpline`
et des `Graph` à deux points, et la réduction de la pression sur les registres.
Ces deux familles représentent environ 86 % des dials OA2024.

Elles restent intéressantes, mais la nouvelle capture donne une priorité claire
pour les petits appels locaux : mesurer la couverture puis éviter de recalculer
les événements inchangés. Une stratégie combinant chemins dense et sparse
répondra mieux aux workloads MCMC et Minuit2 qu'un unique kernel supposé optimal
pour tous les appels.

## Validation acquise et limites

Les comparaisons du benchmark passent pour les quatre configurations CUDA :
écart absolu maximal de LLH `1,88e-12`, et d'histogrammes/erreurs `1,32e-11`.
Les poids individuels OA2024 n'ont pas été matérialisés pour comparaison dans
ce lancement. Les tests dédiés comparent les poids sur de petits modèles.

Les tests CUDA propagation et surfaces ont ensuite passé sur V100, y compris
la propagation compacte et les chemins d'erreur. Les corrections du runner ROOT
et du rejet de `NaN` des callbacks hôtes sont distinctes du compactage.

Aucun changement de défaut de production, gain mesuré sur un fit complet, ni
performance démontrée d'une propagation événementielle sparse n'est revendiqué
ici. Le rapport établit le gain du compactage et propose la prochaine expérience.
