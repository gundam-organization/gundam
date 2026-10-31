# Benchmark résident des poids CUDA sur OA2024

Le point d’entrée est `tests/benchmarks/236CudaWeightBenchmark.py`, avec trois
modes : `benchmark`, `preflight`, `ncu`. Il réutilise le profileur 229 et son
activation à chaud des backends : **un seul chargement des inputs par lancement**,
y compris pour comparer CPU, les deux modes CUDA et les deux layouts de poids.
Il ne lance ni minimisation ni Hesse. Les données sont Asimov ; la sélection des
fichiers et la vraisemblance configurée sont conservées.

## Compiler et mesurer

Dans l’environnement ROOT/Python/CUDA utilisé pour compiler au CC, reconstruire
et installer GUNDAM avec `WITH_PYTHON_INTERFACE=ON`, `WITH_BACKENDS=ON`,
`WITH_CUDA_LIB=ON`, `CMAKE_CUDA_ARCHITECTURES=70` et un build Release.
Pour la correspondance source/instructions dans Nsight, ajouter
`-DCMAKE_CUDA_FLAGS=-lineinfo` (conserver les autres flags CUDA nécessaires).
Ne pas utiliser `-G` pour ces mesures.

Depuis le checkout au CC :

```bash
export GUNDAM_REPO="$PWD"
export PYTHON="$(command -v python3)"
export OA_WORK=/sps/t2k/ablanche/work/results/gundam/GundamInputOA2024
export GUNDAM_MODULE=/sps/t2k/ablanche/install/gundam/lib

sbatch --export=ALL tests/benchmarks/236CudaWeightBenchmark.sbatch \
  --mode benchmark --work-dir "$OA_WORK" --config configOA2024.yaml \
  --override override/v13Prod.yaml --python-module-dir "$GUNDAM_MODULE" \
  --output "$OA_WORK/weights-baseline"
```

Le benchmark compare par défaut CPU, CUDA cached/recompute avec layout
`warpInterleaved`, chacun en référence puis avec descripteurs compacts, dans le
même moteur résident. Les anciens cas `eventMajor` restent disponibles. `--cases` permet
de réduire cette liste. Les scénarios constant/local1/local2/global sont exécutés
après échauffement ; local1 couvre **toutes** les coordonnées libres, même si
leur nombre dépasse `--calls`. `--parameter-match` sélectionne des coordonnées
effectives du fit, éventuellement propres, pas directement les paramètres
physiques avant transformation.

Les sorties reprennent le format du benchmark 229 :

- `report.md` : temps complets et par étage, sans additionner temps hôte et GPU ;
- `<cas>/model.json` : diagnostic CUDA et nouvel objet `weightInventory` ;
- `weightInventory.dialTypes` : effectifs, références événementielles et
  distribution des tailles de payload, par type ;
- tailles réelles des buffers de payload/descripteurs/réponses et distribution
  du nombre de dials par événement ; les octets référencés par les dials peuvent
  compter plusieurs fois une plage de payload partagée ;
- `<cas>/local1-coordinate-costs.json` : classement des coordonnées coûteuses,
  temps de réponses et candidats visités ;
- `comparison.json` : vérification CPU/CUDA des LLH à chaque appel et des
  histogrammes finaux de chaque essai ; ajouter `--event-variables` sans nom de
  variable pour comparer aussi tous les poids et bins finaux, hors chronomètre
  (snapshots volumineux) ;
- `setup.json` par cas : PID commun et `includesDatasetLoading` vrai seulement
  pour le premier cas. Initialisation et activation sont hors chronomètre.

L’inventaire concerne les dials racines ; les nœuds internes de composites ne
sont pas comptés comme des réponses indépendantes. Les compteurs de programmes
existants restent disponibles. L’inventaire impose de reconstruire le module ;
les scripts seuls ne peuvent pas l’obtenir d’une ancienne installation.

## Préparer Nsight Compute

Dans une allocation V100, capturer les versions sans lire les données :

```bash
python3 tests/benchmarks/236CudaWeightBenchmark.py --mode preflight \
  --output "$OA_WORK/weights-preflight"
```

Il faut connaître le chemin/version de `ncu`, la version du driver et l’accès aux
compteurs matériels dans l’allocation. `preflight.json` conserve les versions et
l’environnement ; il ne prouve pas l’autorisation d’accès aux compteurs.
`ERR_NVGPUCTRPERM` pendant la capture nécessite une intervention du support CC.
Ne pas changer les permissions du driver depuis le script.

Choisir un Nsight Compute compatible avec **Volta/V100** : les versions récentes
ne le supportent plus. La [version 2023.3.1](https://docs.nvidia.com/nsight-compute/2023.3/pdf/ReleaseNotes.pdf)
liste GV100 comme supporté ; vérifier aussi sa compatibilité avec le driver du
nœud. Le `ncu` du PATH n’est donc pas nécessairement approprié. Fournir
`--ncu /chemin/vers/ncu` si besoin. Le toolkit servant à compiler peut être
différent de l’installation autonome du profileur.

## Capture ciblée sans recharger le moteur

```bash
sbatch --export=ALL tests/benchmarks/236CudaWeightBenchmark.sbatch \
  --mode ncu --work-dir "$OA_WORK" --config configOA2024.yaml \
  --override override/v13Prod.yaml --python-module-dir "$GUNDAM_MODULE" \
  --output "$OA_WORK/weights-ncu"
```

Par défaut : backends cached/warpInterleaved référence puis compact, scénarios local1 puis global, un
appel capturé par scénario au premier essai, après huit appels d’échauffement.
Tous les autres appels, les chargements, les activations de backend et les
matérialisations restent hors capture. Local1 parcourt toujours toutes les
coordonnées, mais seule la première est capturée par défaut.

Après lecture du classement baseline, utiliser `--profile-call N` pour capturer
la coordonnée d’indice N dans `parameters.json` au cours de local1. Pour un
paramètre isolé, utiliser `--parameter-match 'expression' --scenarios local1`.
Pour étudier exclusivement le recalcul dense :
`--cases CUDA-recomputeResponses-interleaved --scenarios global`.
Pour comparer plusieurs cas dans **la même capture résidente**, passer plusieurs
valeurs à `--cases` ; l’activation à chaud est conservée.

Les gates utilisent `cudaProfilerStart/Stop`, avec `--profile-from-start off` et
un replay **kernel** : aucun replay de l’application ni relecture des inputs.
Si le runtime n’est pas trouvé automatiquement, passer
`--cuda-runtime /chemin/du/toolkit/lib64/libcudart.so` correspondant au build.
Ne pas modifier `CUDA_VISIBLE_DEVICES` défini par Slurm.

Le filtre ne retient que `evaluateResponses`, `evaluateSelectedResponses` et
`reweightEvents`. Le jeu de métriques est `full`, configurable via `--ncu-set`.
Les rapports permettent d’examiner mémoire/transactions/cache, occupation,
registres/spills, instructions et stalls. `--cache-control all` vide les caches
entre les passes ; `--cache-control none` permet une expérience complémentaire,
avec un état de cache moins contrôlé. Le contrôle des fréquences est désactivé
pour ne pas demander de privilèges supplémentaires. Ces choix suivent les
[options de Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightComputeCli/).

Récupérer `weights.ncu-rep`, `command.json`, le log Slurm et `capture/`.
Les logs `NCU capture` indiquent cas/scénario/coordonnées, et les lignes JSON
marquées `hardwareProfiled` identifient les appels capturés. **Ne pas utiliser les
temps de cette exécution pour mesurer les gains** : l’instrumentation et les
replays les perturbent. Les performances se comparent avec `--mode benchmark`,
sans Nsight. Les dossiers existants sont refusés.


## Première variante : descripteurs compacts (opt-in)

La configuration CUDA `compactDials: true` active quatre colonnes de `uint32`
(entrée, offset et taille du payload, référence de politique). Les options
identiques (type, nombre d'entrées, extrapolation, bornes et smoothing) sont
partagées dans une table de politiques, sans changer l'ordre des dials, les
indices des réponses, les payloads ni les multiplications par événement. Les
kernels dense et sparse utilisent les mêmes routines numériques que la
référence ; le cache des entrées transformées est conservé.

Le coût par dial devient 16 octets plus la table de politiques, au lieu du
descripteur générique de 176 octets. Les politiques sont regroupées par leurs
valeurs binaires pour préserver notamment les zéros signés. Les dials Formula,
Composite, surfaces et externes déclenchent un repli du modèle entier vers le
chemin générique, de même qu'un indice ne tenant pas sur 32 bits ou une table
qui annulerait le gain de stockage. Ce repli est explicite dans les diagnostics.
Le benchmark refuse de mesurer un cas nommé `-compact` si ce chemin est inactif.
Le comportement par défaut des applications GUNDAM reste le chemin générique.

Cette itération isole le compactage des descripteurs. La transposition des
payloads et les kernels spécialisés par type restent pour une expérience
ultérieure, selon les performances mesurées sur V100.

Reconstruire **et réinstaller** le module CUDA. Dans une allocation adaptée :

```bash
export GUNDAM_BUILD=/chemin/vers/le/build-cuda
cmake --build "$GUNDAM_BUILD" -j4
cmake --install "$GUNDAM_BUILD"
```

Puis soumettre le benchmark (les tests courts sont lancés dans l'allocation
avant l'unique chargement des données) :

```bash
sbatch --export=ALL "$GUNDAM_REPO/tests/benchmarks/236CudaWeightBenchmark.sbatch" \
  --mode benchmark --work-dir "$OA_WORK" --config configOA2024.yaml \
  --override override/v13Prod.yaml --python-module-dir "$GUNDAM_MODULE" \
  --test-executable "$GUNDAM_BUILD/tests/gundamGTest_cudaBackend.exe" \
  --output "$OA_WORK/benchmark-weights-compact-$(date +%Y%m%d-%H%M%S)"
```

Le filtre des tests préalables couvre `CudaCompactDialModel.*` et
`CudaBackendPropagation.*`, y compris les deux modes et layouts, cache inchangé,
mises à jour sparse/dense, clamping/smoothing et récupération après entrée non
finie. Pour comparer également tous les poids finaux d'OA2024, ajouter
`--event-variables` ; prévoir de l'espace pour les snapshots binaires.

Les nouveaux diagnostics `requestedCompactDials`, `compactDials`,
`compactDialPolicies`, `compactDialFallback` et `deviceDialDescriptorBytes` sont
exportés dans `model.json` et dans le rapport de benchmark. Vérifier le temps
complet, celui des réponses et les coordonnées FSI coûteuses, avant de choisir
un réglage. Aucun gain GPU n'est revendiqué avant ces mesures.

Validation locale de la préparation : compilation C++ hôte des sources CUDA
modifiées et des tests, tests de packing/sémantique sans GPU, tests Python de
capture/CLI et test de rechargement à chaud du benchmark réussis. NVCC et les
tests device restent à exécuter au CC ; le build Mac ne compile pas le `.cu`.
