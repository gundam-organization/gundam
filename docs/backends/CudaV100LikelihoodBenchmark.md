# Vraisemblances statistiques CUDA : préparation V100 au CC Lyon

Les changements CUDA reprennent les principes validés sur MPS, en conservant
les calculs en `double` et les classes `JointProbability` intactes :

- un thread par bin, puis réduction parallèle des contributions par bloc ;
- réduction finale parallèle et déterministe, sans atomiques ;
- réutilisation des poids, histogrammes et LLH quand les dépendances de la
  prédiction n'ont pas changé ;
- toujours un seul `double` de LLH stat rapatrié pour Poisson et Barlow 2022.

Une génération de poids externes, une mise à jour des observables/binning,
une invalidation après erreur ou un cache non initialisé imposent un nouveau
calcul. `recomputeResponses` continue à tout recalculer. Les autres
vraisemblances conservent leur chemin CPU.

Le noyau CUDA avait déjà un thread par bin. La nouvelle politique automatique
choisit des blocs de 64 threads jusqu'à 8 192 bins du modèle, puis 256 au-delà.
C'est **une hypothèse de départ à mesurer**, pas un réglage validé sur V100.
La réduction finale utilise un bloc de 256 threads. CUDA ne reprend pas
l'arithmétique à deux floats de Metal.

## Compiler dans une allocation GPU

Charger un environnement ROOT/Python cohérent et un toolkit CUDA 12.x. CUDA 13
ne compile plus pour Volta ; cette limitation est documentée dans les
[notes NVIDIA](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html#deprecated-architectures).
Sélectionner explicitement l'architecture V100, `70` :

```sh
cmake -S . -B cmake-build-v100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_BACKENDS=ON -DWITH_CUDA_LIB=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DWITH_PYTHON_INTERFACE=ON -DWITH_TESTS=ON
cmake --build cmake-build-v100 -j4
ctest --test-dir cmake-build-v100 -L cuda --output-on-failure
```

Exécuter compilation et tests dans l'allocation, pas sur le serveur d'accueil.
Utiliser le même environnement pour compiler, soumettre et lancer les workers.
La version de Python doit correspondre au module GUNDAM compilé.

## Lancer le benchmark

`tests/benchmarks/235CudaV100LikelihoodBenchmark.py` orchestre le profileur
résident existant, avec des expériences indépendantes et des comparaisons
numériques entre réglages. Par défaut :

- CPU, CUDA avec cache et CUDA sans cache de réponses ;
- tailles de blocs `auto` et `256` ;
- trois répétitions, après échauffement ;
- scénarios constant, un paramètre, deux paramètres et tous les paramètres ;
- vraisemblance configurée dans les inputs, sans changement de physique ;
- vérification du GPU réellement sélectionné : son nom doit contenir `V100` ;
- LLH device obligatoire et rapatriement de exactement 8 octets pour le total,
  sans poids, état d'événements ni histogrammes pendant les appels chronométrés.

Le mode `resident`, par défaut, charge les inputs une fois par vraisemblance pour tout le balayage de réglages, puis
active successivement CPU et CUDA depuis les événements déjà en mémoire. Les
observables et le binning dynamique/lissé sont supportés. Le mode `isolated`
recharge les inputs dans un processus indépendant par backend. Les temps
d'initialisation et de changement de backend sont exclus dans les deux cas.

Exemple Upgrade, dans une allocation V100, depuis le checkout au CC :

```sh
python3 tests/benchmarks/235CudaV100LikelihoodBenchmark.py \
  --work-dir /chemin/upgrade-cc-2026 --config ndUpCcInc.yaml \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml \
  --python-module-dir "$PWD/cmake-build-v100/src/PythonInterface" \
  --test-executable "$PWD/cmake-build-v100/tests/gundamGTest_cudaBackend.exe" \
  --output "$PWD/work/v100-llh-upgrade"
```

Les tests device/LLH sont alors exécutés avant le benchmark. Pour OA2024,
remplacer les chemins/configuration par ceux réellement présents au CC, par
exemple `--config configOA2024.yaml --override override/v13ProdFewRuns.yaml`.
Aucune sélection de fichiers ni limite d'événements n'est ajoutée par le script.

Pour explorer davantage de réglages, ajouter :

```sh
--block-sizes auto 32 64 128 256 --reuse-modes on off
```

`reuse off` désactive la réutilisation de la prédiction ; le cache de réponses
reste actif dans le mode `cachedResponses`. Cela permet d'isoler son coût des
poids/histogrammes/LLH. Tous les réglages utilisent la nouvelle réduction
parallèle : ce n'est pas une comparaison avec l'ancien noyau final séquentiel.

Pour tester explicitement les deux fonctions :

```sh
--likelihoods PoissonLLH BarlowLLH_BANFF_OA2021
```

Pour un seul paramètre libre ou une sélection `--parameter-match` étroite,
utiliser `--scenarios constant local1 global` pour exclure `local2`.

Ces options changent explicitement la vraisemblance du benchmark. Barlow
conserve les options Barlow reconnues ; Poisson active
`ignoreBinsWithZeroPredictionAtPrior` pour éviter des contributions infinies
sur les bins vides au prior. Sans cette option, la configuration d'origine est
conservée et doit déjà sélectionner une des deux fonctions portables.

Chaque combinaison recharge les inputs en mode isolé : pour une exploration
large, adapter le temps Slurm et commencer par un balayage réduit. Les séquences
de coordonnées sont identiques pour CPU/CUDA, les tailles de bloc et les modes
avec/sans réutilisation. Les poids/histogrammes sont matérialisés hors timer pour
les vérifications. Les variations utilisent les coordonnées effectives du fit,
avec contrôle du domaine ; ce n'est pas un rejeu exact de Migrad.

## Soumission batch

Le modèle `tests/benchmarks/235CudaV100LikelihoodBenchmark.sbatch` demande
un GPU, quatre CPU et 48 Gio de RAM sur `gpu_v100`, pendant deux heures.
Le CC présente cette partition dans sa [documentation Slurm](https://doc.cc.in2p3.fr/en/Computing/slurm/configuration.html)
et demande de vérifier l'état et les droits courants avec `sinfo` / `scontrol`.
Les ressources et la durée du modèle se surchargent avec les options de `sbatch`.

Après activation de l'environnement de compilation :

```sh
export GUNDAM_REPO="$PWD"
export PYTHON="$(command -v python3)"
sbatch --export=ALL tests/benchmarks/235CudaV100LikelihoodBenchmark.sbatch \
  --work-dir /chemin/upgrade-cc-2026 --config ndUpCcInc.yaml \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml \
  --python-module-dir "$GUNDAM_REPO/cmake-build-v100/src/PythonInterface" \
  --test-executable "$GUNDAM_REPO/cmake-build-v100/tests/gundamGTest_cudaBackend.exe" \
  --output "$GUNDAM_REPO/work/v100-llh-upgrade"
```

La sélection Slurm du GPU est respectée : ne pas remplacer son
`CUDA_VISIBLE_DEVICES` par un numéro physique. `--device-index 0` désigne le
premier GPU visible du processus. `--allow-other-gpu` permet d'utiliser le même
script sur un autre GPU, avec son nom conservé dans les diagnostics.

## Résultats à examiner

Dans le répertoire de sortie :

- `report.md` : temps par évaluation, temps GPU de la LLH, octets rapatriés ;
- `summary.json` : état de complétion et expériences exécutées ;
- sous chaque expérience : `comparison.json`, puis `matrix-comparison.json`
  pour les expériences suivantes, contrôlent les totaux **et la LLH stat**,
  les histogrammes/erreurs et les points de paramètres ;
- fichiers par répétition et scénario : échantillons bruts et timings ;
- `settings.json` de chaque expérience : commit, GPU/driver via `nvidia-smi`, environnement Slurm et
  affinité CPU ; manifests : module Python et son hash, configuration résolue,
  taille de bloc réellement choisie ;
- `local1-coordinate-costs.json` : coordonnées les plus coûteuses ;
- `device-tests.log` et logs des workers : détails en cas d'échec.

Un écart numérique ou un transfert inattendu arrête le script avec un code non
nul. Les dossiers existants ne sont pas écrasés. Les intervalles GPU recouvrent
le temps d'évaluation CPU : ne pas les additionner. Les modes et réglages sont
exécutés séquentiellement ; examiner aussi les distributions et la variabilité
thermique avant de retenir une taille de bloc.

## Validation effectuée sur Mac

Build Release sans CUDA réussi, 245 tests rapides C++ réussis. Les tests C++
spécifiques CUDA ont aussi été compilés comme code hôte pour vérifier leur
syntaxe, sans exécution GPU. Le nouveau script a passé un smoke test réel sur
CPU avec huit combinaisons de réglages et les deux vraisemblances ; une
altération de la seule LLH stat est bien rejetée. La régression du profileur
partagé avec le benchmark de migration passe également.

La compilation NVCC, les nouveaux kernels et les performances V100 restent
**à valider au CC**. Aucun gain CUDA chiffré n'est revendiqué ici.

Les logs de validation locale sont conservés dans
`cmake-build-release/benchmarks/CudaV100Preparation-20261030/`.
