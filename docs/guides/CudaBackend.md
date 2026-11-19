# Backend CUDA : dials, observables et binning

`Backends::CudaBackend` est enregistré dans `BackendFactory` sous les noms
`CUDA` et `cuda`. Son interface publique ne dépend pas du toolkit. Sans CUDA,
la construction/configuration reste possible, mais l'initialisation explique
qu'il faut compiler avec `WITH_CUDA_LIB=ON` et un toolkit utilisable.

Le premier exécuteur numérique a passé les cinq tests dédiés au CC sur V100
(job `62374030`, 24 octobre 2026), dont les comparaisons CPU/CUDA dans les deux
modes. Cette validation porte sur les modèles synthétiques de la suite ; un fit
complet et un modèle de production restent à tester.

## Configuration

```yaml
backendManagerConfig:
  isEnabled: true
  useBackend: CUDA
  options:
    - name: CUDA
      deviceIndex: 0
      propagationMode: cachedResponses
      weightLayout: warpInterleaved
```

`propagationMode: cachedResponses` et `weightLayout: warpInterleaved` sont
les valeurs par défaut. Le rangement entrelacé revient à `eventMajor` si le
padding dépasse deux fois la taille initiale des indices.

`deviceIndex` vaut zéro par défaut et désigne un GPU visible par le processus.
Respecter `CUDA_VISIBLE_DEVICES` fourni par Slurm : avec un seul GPU alloué,
utiliser zéro. Les modes suivent la même convention que MPS :

| Mode | Réponses des dials | Poids et histogrammes |
|---|---|---|
| `cachedResponses` (défaut) | Une réponse par dial unique, conservée sur GPU ; recalcul si l'entrée transformée change. | Recalculés à chaque propagation. |
| `recomputeResponses` | Tous les dials uniques sont réévalués à chaque propagation. | Recalculés à chaque propagation. |

Les deux modes utilisent le même exécuteur en double précision native. Le
monitor indique `CUDA/cachedResponses` ou `CUDA/recomputeResponses`. Aucun
repli vers le calcul CPU n'est effectué.

## Chaîne de calcul

`build()` copie le modèle ; `buildShared()` conserve le modèle partagé. La
préparation valide les indices et payloads, puis charge les descripteurs, les
références de dials et les payloads dans des buffers CUDA persistants. Un index
par bin contient les slots de résultats des événements qui y contribuent.
L'ordre des métadonnées des événements peut différer de celui de ces slots.

Chaque propagation transfère les paramètres, évalue les réponses nécessaires,
multiplie les dials dans l'ordre des références de chaque événement, applique
le cap à ce produit puis multiplie par le poids initial. Les kernels réutilisent
les fonctions numériques communes pour le mirroring, l'interpolation,
l'extrapolation, les bornes et le smoothing des réponses.

Une réduction par bloc calcule les sommes et sommes des carrés de chaque bin.
Elle utilise un arbre fixe en double précision, sans atomiques flottantes ;
l'ordre des additions peut différer de CPU. Les événements hors histogramme
restent dans les poids de sortie mais ne contribuent à aucun bin.

La vraisemblance est évaluée sur l'hôte par la routine commune, après lecture
des histogrammes. Sans définition de vraisemblance, les poids et histogrammes
restent disponibles mais `getLikelihood()` lève une erreur.

## Exécution et résultats

- Le contexte et son stream non bloquant sont privés au backend. Les opérations
  sélectionnent le GPU puis restaurent le périphérique courant du thread.
  Aucun `cudaDeviceReset()` global n'est effectué.
- `requestPropagation()` synchronise son stream avant de retourner un token
  valide. `isReady()` et `wait()` reflètent cette exécution synchrone.
- Les poids restent sur GPU jusqu'à leur matérialisation ou leur premier accès
  hôte. Les histogrammes sont lus pour la vraisemblance, sinon à la demande.
  Une seconde lecture ne provoque pas de transfert supplémentaire.
- Chaque demande invalide les tokens précédents, même si elle échoue. Un rebuild
  ou une reconfiguration les invalide aussi ; les identifiants ne sont pas
  réutilisés après reconfiguration.
- Les entrées non finies, les résultats GPU non finis et les erreurs CUDA
  empêchent la publication d'un résultat. Un échec invalide le cache de réponses
  pour la tentative suivante. Une erreur CUDA fatale peut nécessiter de recréer
  le processus ; la reprise numérique ne répare pas un contexte matériel perdu.
- Les temps des étapes GPU viennent d'événements CUDA. Les durées de build,
  lecture et vraisemblance sont mesurées sur l'hôte. Les totaux couvrent les
  propagations réussies depuis le dernier build.

## Périmètre

Sont implémentés : `Norm`, `Shift`, `CompactSpline`, `UniformSpline`,
`MonotonicSpline`, `GeneralSpline`, `Graph`, `Surface` (`Bilinear`, `Bicubic`),
`ExternalWeightDispatcher`, `Tabulated`, `Kriged`, les entrées paramètre 1D/2D, les
histogrammes fixes/dynamiques stricts ou lissés et la vraisemblance totale. Les payloads doivent être valides
et finis, les coordonnées des graphes/splines générales strictement croissantes.

Les surfaces concernent les poids et les réponses d’observables. Leur cache utilise les deux
entrées transformées et des groupes de dépendances disjoints pour éviter les
écritures concurrentes. La compilation NVCC et les trois suites CUDA, dont
`cuda-backend-surfaces`, ont été validées au CC le 24 octobre 2026.

Les entrées de dial provenant d'une variable d'événement sont refusées
pendant la préparation. Les modifications d'observables fournies à la
propagation sont acceptées. La sortie `SampleLikelihoods` n'est pas implémentée.
`EventState` est disponible pour les modèles avec observables, binning dynamique strict ou binning lissé.

## Réponses d’observables

`applyToVariable` compose `Override`, les produits puis les sommes, à partir
de la référence initiale. Les réponses de dials sont partagées avec les poids.
Les valeurs et bins restent sur le GPU ; `EventState` permet leur lecture et la
remise à jour des événements et listes de bins sur l’hôte via `BackendManager`.
Le binning strict dynamique et les réponses d’observables ont passé les cinq
suites CUDA au CC le 25 octobre 2026 (1,53 s au total). Le binning lissé et les
contributions hors bins sont maintenant implémentés ; leur exécution GPU reste
à valider. Voir [le protocole du binning lissé](../backends/CudaSmoothedBinning.md).

Les tests hôte, CPU/MPS et Python passent localement. Voir le
[protocole de validation](../backends/CudaObservableResponses.md).

## Poids externes et tables

Les producteurs `ExternalWeightDispatcher` et les mises à jour des tables
Tabulated/Kriged restent sur l'hôte. CUDA charge les blocs dont la génération
change, en doubles, puis évalue les réponses concernées sur le GPU. Les groupes
par bloc sont disjoints et indépendants des changements de paramètres.
Les limites et le lissage des réponses utilisent la sémantique commune.
Les coefficients, indices et leur ordre sont conservés, y compris les termes
Kriged signés, répétés ou non triés et les stencils vides.

`cachedResponses` conserve les réponses tant que leur génération ne change pas ;
`recomputeResponses` les recalcule à chaque appel tout en conservant les blocs
inchangés sur le GPU. Après une erreur ou une reconstruction, le prochain appel
recharge les blocs et réamorce toutes les réponses. Les pointeurs hôte sont
consommés avant le retour de `requestPropagation`.

Les tables actuelles n'exposent pas de révision de leur producteur :
`BackendManager` publie une génération à chaque propagation, même à paramètres
identiques. Elles sont donc transférées à chaque appel. Les dispatchers ont une
vraie génération de publication et permettent les transferts sélectifs.

Les validations hôte et les régressions Python CPU/MPS passent localement.
L’exécution CUDA des tests externes/tables a été confirmée au CC.
Pour les relancer dans une allocation GPU :

```bash
cmake --build /sps/t2k/ablanche/build/gundam \
  --target gundamGTest_cudaBackend.exe GUNDAM -j4 &&
ctest --test-dir /sps/t2k/ablanche/build/gundam \
  -R '^cuda-backend-' --no-tests=error --output-on-failure
```

`cuda-backend-external` compare les deux modes et les deux rangements à CPU,
les tables aux dials natifs et les dispatchers à un vrai producteur. Il vérifie
les transferts sélectifs, les générations initialement non publiées, les blocs
vides, les erreurs, la reprise et la reconstruction.

Les tests Python existants proposent un opt-in pour tester aussi le parcours
complet dans les deux modes CUDA. Sans cet opt-in, ils continuent à fonctionner
sans CUDA. Une demande explicite échoue si CUDA n'est pas utilisable.
Depuis la racine du dépôt au CC (build avec `WITH_PYTHON_INTERFACE=ON`) :

```bash
repoDir="$PWD"
export PYTHONPATH="/sps/t2k/ablanche/build/gundam/src/PythonInterface${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p cuda-external-tables
cd cuda-external-tables
bash "$repoDir/tests/fast-tests/005PythonSetup.sh" "$repoDir/tests/fast-tests" &&
GUNDAM_TEST_CUDA=1 venv/bin/python "$repoDir/tests/fast-tests/214ExternalWeightBackend.py" &&
GUNDAM_TEST_CUDA=1 venv/bin/python "$repoDir/tests/fast-tests/215CpuTableDials.py"
```

L'interpréteur utilisé pour créer `venv` doit correspondre à celui du module
Python compilé. Les logs et les fichiers de test restent dans ce répertoire.

## Sources

- `Implementations/CUDA/include/CudaBackend.h` : interface et mode de propagation.
- `Implementations/CUDA/src/CudaBackend.cpp` : configuration, tokens,
  matérialisation et vraisemblance sur l'hôte.
- `Implementations/CUDA/include/CudaWeightModel.h` et `src/CudaWeightModel.cpp` :
  préparation et validation du modèle, testables sans CUDA.
- `Implementations/CUDA/src/CudaDeviceContext.cu` : buffers, stream, cache et
  kernels ; réutilise `Semantics/BackendDialSemanticsCore.h`, sans header ROOT.

## Validation

Le squelette initial a passé `CudaBackendDevice.BuildsSharedModelAndRestoresDeviceSelection`
le 24 octobre 2026 au CC : job `62373853`, nœud `ccwgslurm0101`,
Tesla V100-SXM2-32GB, 142 ms. Un seul GPU était visible ; le changement entre
deux périphériques distincts n'est pas validé. Le smoke test autonome FP64
avait aussi réussi en interactif et en batch avec CUDA 12.4.

La suite locale comporte les tests `CudaBackend.*` et `CudaWeightModel.*`.
Le job `62374030`, exécuté sur le même V100, a passé les cinq tests dédiés en
184 ms : un test de cycle de vie (163 ms) et quatre tests de propagation
(20 ms au total). Ces durées décrivent la suite et ne constituent pas un
benchmark de propagation.

Les tests GPU `CudaBackendPropagation.*` ont comparé CPU/CUDA pour les deux modes : les sept types de dials scalaires, répétitions et changements
de paramètres, frontières, mirroring, extrapolation, limites lissées et cap.
Ils couvrent aussi les poids négatifs/nuls, un nombre d'événements non multiple
de la taille de bloc, les slots réordonnés, bins vides et événements hors bins,
les lectures à la demande, les tokens obsolètes, les débordements et la reprise
après échec. Le test de cycle de vie initial passe également avec ce nouvel
exécuteur. La comparaison principale utilise 521 événements et huit jeux de
paramètres pour chacune des huit combinaisons mode/cap/extrapolation. Les
poids et histogrammes sont comparés avec une tolérance de
`2e-12 * max(1, abs(référence))`, les vraisemblances avec
`2e-11 * max(1, abs(référence))`. Le multi-GPU et H100 restent à valider.

Un [regular-test de fit et covariance](../backends/BackendFitCovariance.md)
est aussi disponible. Il ajuste trois paramètres corrélés sur 540 événements,
puis compare les valeurs ajustées et toute la matrice Hesse, y compris ses
corrélations et valeurs propres, à CPU et à une référence analytique. Le test
CPU/original passe localement ; les comparaisons CUDA dans les deux modes ont
été validées au CC. Les builds sans backend/CUDA et les machines sans GPU ignorent ces tests
proprement, sauf si `GUNDAM_REQUIRE_CUDA_TESTS=1` impose leur exécution.

Depuis la racine du dépôt, charger ROOT et le compilateur hôte habituels puis
compiler dans une allocation interactive avec le toolkit utilisateur :

```bash
cmake -S . -B /sps/t2k/ablanche/build/gundam \
  -DWITH_CUDA_LIB=ON -DWITH_BACKENDS=ON \
  -DWITH_TESTS=ON -DWITH_GOOGLE_TEST=ON \
  -DCMAKE_CUDA_COMPILER=/sps/t2k/ablanche/install/cuda-12.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 &&
cmake --build /sps/t2k/ablanche/build/gundam \
  --target gundamGTest_cudaBackend.exe -j4 &&
ctest --test-dir /sps/t2k/ablanche/build/gundam \
  -R '^cuda-backend-' --output-on-failure
```

La construction ne lance aucun test GPU. Les cibles embarquent le chemin du
runtime dans leur RPATH. Pour soumettre les suites en batch :

```bash
sbatch tests/cuda/CcLyonSmoke.sbatch \
  /sps/t2k/ablanche/build/gundam/tests gundamGTest_cudaBackend.exe
```

Le smoke test autonome reste disponible sans ROOT :

```bash
sbatch tests/cuda/CcLyonSmoke.sbatch build-cuda-smoke-12.4
```
