# Développer CUDA depuis macOS avec le CC-IN2P3

Documentation vérifiée le 24 octobre 2026. Le smoke test a été validé sur un
V100 au CC, en interactif et en batch, avec un toolkit CUDA 12.4 utilisateur.
Le test du squelette de backend a également réussi en batch sur V100
(job Slurm `62373853`, 142 ms). Le premier exécuteur numérique CUDA a également passé les cinq tests dédiés
sur V100 : job `62374030`, 184 ms pour la suite complète, comparaisons CPU/CUDA
incluses dans les deux modes. Un fit complet et un modèle de production restent
à tester.

## État de GUNDAM

L'ancien `CacheManager` possède déjà des sources CUDA. La bibliothèque
`Backends` propose CPU, MPS et CUDA, sélectionnable avec `type: CUDA`.
CUDA implémente les dials 1D, les histogrammes à binning fixe et la vraisemblance
sur l'hôte, avec les modes `cachedResponses` et `recomputeResponses`. Les tests
numériques des deux modes ont réussi sur V100 ; voir le périmètre de validation
dans le guide du backend.
Voir [le backend CUDA](CudaBackend.md) pour sa configuration et ses tests.

`WITH_CUDA_LIB` accepte :

| Valeur | Comportement |
| --- | --- |
| `OFF` (défaut conservé) | Aucune recherche de CUDA. |
| `AUTO` | Active CUDA si CMake trouve un compilateur utilisable ; sinon continue sans CUDA. |
| `ON` | Exige CUDA et arrête la configuration si le compilateur manque. |

Un toolkit trouvé mais incomplet ou incompatible peut encore faire échouer la
configuration en mode `AUTO`. Utiliser `OFF` pour exclure explicitement CUDA.
Un build CUDA demande CMake ≥ 3.18 et un compilateur hôte compatible avec le
toolkit et le ROOT utilisé. Le standard CUDA suit celui de ROOT dans GUNDAM.
L'activation concerne aussi l'ancien `CacheManager` lorsqu'il est compilé.

`GUNDAM_CUDA_ENABLED` indique le résultat de la détection aux CMakeLists ; les
nouveaux fichiers `.cu` devront être ajoutés à `GundamBackends` sous cette
condition. La cible est déjà liée au runtime `CUDA::cudart` dans ce cas.

## Compiler au CC

Le CC documente la connexion via `cca.in2p3.fr`. La compilation parallèle doit
se faire dans une allocation, pas sur les serveurs d'accueil.
[Connexion et règles des serveurs d'accueil](https://doc.cc.in2p3.fr/en/Daily-usage/cca-connect.html).

Depuis le Mac :

```bash
ssh votre_login@cca.in2p3.fr
```

Depuis le serveur d'accueil, demander une session GPU interactive :

```bash
srun -p gpu_v100_interactive --gpus=1 --ntasks=1 \
  --cpus-per-task=4 --mem=8G --time=01:00:00 --pty bash -i
```

Pour un H100, remplacer la partition par `gpu_h100_interactive`. L'accès GPU
dépend des ressources accordées au groupe.
[Partitions](https://doc.cc.in2p3.fr/fr/Computing/slurm/configuration.html),
[syntaxes Slurm](https://doc.cc.in2p3.fr/fr/Computing/slurm/use-syntax.html).

Dans cette session :

```bash
hostname
nvidia-smi
command -v nvcc
ls -ld /usr/local/cuda* 2>/dev/null
module avail cuda
module view cuda
```

Ne pas supposer que `/usr/local/cuda-12/bin/nvcc` existe : le chemin donné
par la documentation du CC peut différer de l'installation réelle. Si `nvcc`
n'est pas dans le PATH, charger un module CUDA effectivement listé (avec son
nom complet), ou ajouter au PATH le répertoire `bin` d'un toolkit installé.
`module avail/view cuda` sert à chercher les modules ; leur présence n'est pas
garantie. Au besoin, chercher les installations système :

```bash
find /usr/local /opt -maxdepth 5 -type f -name nvcc -print 2>/dev/null
```

Après avoir activé le toolkit :

```bash
command -v nvcc
nvcc --version
```

[Gestion des modules au CC](https://doc.cc.in2p3.fr/fr/Software/modules.html).
La ligne « CUDA Version » de `nvidia-smi` indique la compatibilité du pilote,
pas nécessairement la version du toolkit installé : vérifier `nvcc --version`.
Pour un toolkit particulier, le CC propose une installation dans le stockage
du groupe ou une image Apptainer. Charger le même environnement pour compiler
et exécuter.
[Environnement CUDA du CC](https://doc.cc.in2p3.fr/en/Computing/slurm/hpc-env.html).

Choisir explicitement les architectures :

| GPU | `CMAKE_CUDA_ARCHITECTURES` | Toolkit |
| --- | --- | --- |
| V100 | `70` | CUDA 12.x |
| H100 | `90` | CUDA ≥ 11.8, avec pilote compatible |
| V100 et H100 | `70;90` | CUDA 11.8 ou 12.x |

Les V100 utilisent Volta (7.0), les H100 Hopper (9.0).
[Architectures NVIDIA](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).
CUDA 13 a supprimé la compilation pour Volta : utiliser CUDA 12.x pour un build
destiné aux V100.
[Notes CUDA 13](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html).

## Premier test, sans installer ROOT

Depuis la racine d'une copie du dépôt sur un stockage partagé du CC, dans la
session interactive V100 :

```bash
# nvcc doit maintenant être dans le PATH ; vérifier qu'il s'agit de CUDA 12.x.
# Exécuter le bloc dans un sous-shell pour arrêter au premier échec.
(
  set -e
  cudaCompiler=$(command -v nvcc)
  "$cudaCompiler" --version
  cmake -U CMAKE_CUDA_COMPILER -S tests/cuda -B build-cuda-smoke \
    -DCMAKE_CUDA_COMPILER="$cudaCompiler" \
    -DCMAKE_CUDA_ARCHITECTURES=70
  cmake --build build-cuda-smoke -j4
  ctest --test-dir build-cuda-smoke -R '^cuda-smoke$' --output-on-failure
)
```

Ce test affiche le GPU, alloue de la mémoire, lance un kernel FP64, synchronise
le GPU et vérifie les 257 valeurs rapatriées. L'absence de GPU est un échec,
afin qu'un job mal configuré ne soit pas annoncé comme réussi.

Quitter ensuite l'allocation interactive, revenir à la racine du dépôt et
soumettre le même binaire en batch :

```bash
exit
sbatch tests/cuda/CcLyonSmoke.sbatch build-cuda-smoke
squeue -u "$USER"
# Remplacer 123456 par l'identifiant renvoyé par sbatch :
sacct -j 123456 --format=JobID,State,ExitCode,Elapsed
cat gundam-cuda-smoke-123456.log
```

Pour H100, compiler avec `90` et soumettre avec
`sbatch -p gpu_h100 tests/cuda/CcLyonSmoke.sbatch build-cuda-smoke`.
Le script accepte le répertoire de build comme premier argument ; il doit
être visible depuis le nœud batch. Adapter son environnement si un toolkit
non système est utilisé. Ne pas écraser `CUDA_VISIBLE_DEVICES` défini par Slurm.

## Compiler GUNDAM

Après avoir chargé votre environnement Linux habituel (ROOT avec Minuit2,
CMake et compilateur hôte), dans une allocation interactive :

```bash
cmake -S . -B build-cuda \
  -DWITH_CUDA_LIB=ON -DWITH_BACKENDS=ON -DWITH_TESTS=ON \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  '-DCMAKE_CUDA_ARCHITECTURES=70;90'
cmake --build build-cuda -j4
ctest --test-dir build-cuda -R '^cuda-smoke$' --output-on-failure
```

Le test de ce build se trouve dans `build-cuda/tests/cuda/`. On peut aussi le
soumettre avec `sbatch tests/cuda/CcLyonSmoke.sbatch build-cuda/tests/cuda`.
Il valide l'infrastructure CUDA. Les comparaisons CPU/CUDA des poids,
histogrammes et vraisemblances sont dans `cuda-backend-propagation`, disponible
avec le build GUNDAM complet et GoogleTest. Voir [la validation CUDA](CudaBackend.md#validation).

La configuration et la compilation ne lancent pas ce test GPU. Le toolkit
suffit pour compiler sur Linux sans GPU, à condition de préciser l'architecture.
Éviter `native` pour un build sans GPU ou destiné à plusieurs types de cartes.
[Architectures CMake](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html).

## Travailler depuis le Mac

Ce Mac est un Apple Silicon (`arm64`). NVIDIA a arrêté le support macOS après
CUDA 10.2 ; ce toolkit historique ne fournit pas une chaîne de compilation
CUDA moderne pour Apple Silicon. Copier les headers peut aider à consulter
l'API, mais ne permet pas de lier ni de valider une application CUDA macOS.
[Fin du support macOS](https://docs.nvidia.com/cuda/archive/10.2/cuda-toolkit-release-notes/).

Le workflow conseillé est :

1. Éditer localement dans CLion et conserver le profil Mac CPU/MPS avec
   `-DWITH_CUDA_LIB=OFF` ou `AUTO`.
2. Synchroniser les sources vers le CC via Git ou rsync/SSH, en excluant les
   répertoires de build Mac. Ne jamais partager un cache CMake entre Mac et Linux.
3. Compiler dans une allocation interactive et lancer les tests courts dessus.
4. Soumettre les tests plus longs avec `sbatch`, puis récupérer les logs.

CLion propose une toolchain « Remote Host » et un profil CMake associé, avec
sources locales et synchronisation distante.
[Documentation CLion](https://www.jetbrains.com/help/clion/remote-projects-support.html).
Une connexion SSH CLion à `cca` ne réserve aucun GPU : les compilations
parallèles et les exécutions doivent toujours être placées dans Slurm. Pour
commencer, garder CLion local et piloter les commandes Slurm depuis un terminal
SSH séparé est le montage le plus simple. Une connexion IDE directe à un nœud
alloué dépend des règles SSH du CC et de la durée de l'allocation ; elle n'est
pas supposée disponible par ces scripts.

Une VM/conteneur Linux disposant d'un toolkit compatible peut servir à des
compilations sans GPU. Cela produit des binaires Linux, et ne remplace pas les
tests CUDA sur un GPU NVIDIA. Depuis ce Mac, le CC évite également de gérer
localement les différences ARM/x86 et les dépendances ROOT Linux.
