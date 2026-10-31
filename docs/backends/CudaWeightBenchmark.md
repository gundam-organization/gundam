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

Le benchmark compare par défaut CPU, CUDA cached/recompute, chacun avec
`eventMajor` et `warpInterleaved`, dans le même moteur résident. `--cases` permet
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

Par défaut : backend cached/warpInterleaved, scénarios local1 puis global, un
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
