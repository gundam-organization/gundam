## DatasetDefinition

[< back to parent (Propagator)](Propagator.md)

### Description

Defines how the data should be loaded to the samples.
Each config entry refers to a data set.

### Config options

As an entry list

| Option                               | Type   | Description                                                         | Default |
|--------------------------------------|--------|---------------------------------------------------------------------|---------|
| name                                 | string | Name of the dataset                                                 |         |
| mc (see bellow)                      | json   | Config for the MC entry                                             |         |
| data (see bellow)                    | json   | Config list for the Data entries                                    |         |
| selectedDataEntry                    | string | Name of the 'data' entry that should be used to load data events    | Asimov  |
| selectedToyEntry                     | string | Name of the 'data' entry that should be used to perform a toy fit   | Asimov  |
| isEnabled                            | bool   | Specify if it should be considered during the runtime               | true    |
| showSelectedEventCount               | bool   | Show the number of events passing the selection cut for each sample | true    |
| nbMaxThreadsForLoad                  | int    | Maximum loading readers, also bounded by the global CPU thread count | 4       |
| devSingleThreadEventSelection        | bool   | Force the event selection to be performed in single thread          | false   |
| devSingleThreadEventLoaderAndIndexer | bool   | Force the event loading to be performed in single thread            | false   |

Before event selection, the loader sums the sizes of enabled main and friend files,
counting each physical file once. Inputs below 500 MB (500,000,000 bytes) use one
reader. Larger inputs run a raw, buffered file-read benchmark without ROOT
decompression, comparing 1, 2, 4, ... readers up to the configured limit (including
the limit itself). The smallest count reaching at least 90% of the best measured
aggregate throughput is used for selection and loading. Explicit single-thread
development options still take precedence.

Each benchmark configuration reads the same total volume, distributed across all
enabled physical files in proportion to their sizes, including friends. It starts
at 16 MiB and can repeat at 32 and 64 MiB when measurements are shorter than
200 ms. Further rounds stop once the total elapsed time reaches two seconds;
an individual read or round can take longer on slow storage. With a four-reader
limit the maximum total read volume is 336 MiB. Trials use distinct file regions,
but existing operating-system caches can still influence the result. Unknown file
sizes or read failures retain the configured thread limit; no benchmark runs when
only one reader is allowed or both development options force a single thread.


#### mc

| Option                  | Type                         | Description                                                                 | Default |
|-------------------------|------------------------------|-----------------------------------------------------------------------------|---------|
| tree                    | string                       | Name of the TTree containing the data in each file                          |         |
| selectionCutFormula     | formula                      | Global selection cut (should return 0 if not selected)                      |         |
| nominalTreeWeightFormula | formula                     | Formula that returns the base weight of a given event                       |         |
| filePathList            | list(string \| json)          | list of ROOT files containing the TTree; entries can attach friend TTrees   |         |
| additionalLeavesStorage | list(string)                 | list of variables to be stored in memory                                    |         |
| variablesTransform      | list(json)                   | Deprecated. Use `variableDict` entries with `evalFromLib` instead.          |         |
| variableDict            | list(json)                   | dictionary translating a leaf, formula, or shared-library output to a variable name |         |
| fromHistContent         | json                         | use hist bin content directly. This will create dummy events                |         |

Formula fields can be written as a single expression, a list of expressions, or a list of named expression entries:

```yaml
myFormula: "[var1] * 4"
myFormula: ["[var1] * 4", "[var2] + 1"]
myFormula:
  - { name: "part1", expr: "[var1] * 4" }
  - { name: "part2", expr: "[var2] + 1" }
```

`variableDict` entries can define either an `expr` or an `evalFromLib` block:

```yaml
variableDict:
  - { name: "myVar", expr: "treeBranch + 10." }
  - name: "myVarFromLib"
    evalFromLib:
      title: "optional title"
      libraryFile: "foo.so"
      inputList:
        - "[myVar]"
        - "TMath::Abs(mode)"
      messageOnError: "optional build hint"
```

`filePathList` keeps its historical list-of-strings syntax:

```yaml
filePathList:
  - "${DATA_DIR}/events.root"
```

It can also use named entries to attach ROOT friend trees. The `name` keys identify list entries for configuration overrides; friend names are ROOT aliases and should be used to qualify their branches in formulas.

```yaml
filePathList:
  - name: "run1"
    path: "${DATA_DIR}/events-run1.root:events"
    friendList:
      - name: "weights"
        path: "${DATA_DIR}/weights-run1.root:weights"
      - name: "truth"
        path: "${DATA_DIR}/truth-run1.root:truth"
  - name: "run2"
    path: "${DATA_DIR}/events-run2.root:events"
    isEnabled: true
    friendList:
      - name: "weights"
        path: "${DATA_DIR}/weights-run2.root:weights"
        isEnabled: true
      - name: "truth"
        path: "${DATA_DIR}/truth-run2.root:truth"

variableDict:
  - { name: "weight", expr: "weights.nominalWeight" }
```

To attach several friend trees from the same ROOT file, use `treeList`:

```yaml
friendList:
  - name: "xsec_syst_friends"
    path: "/path/to/friend/trees/file.root"
    treeList: ["friendTree1", "friendTree2"]

variableDict:
  - { name: "weight", expr: "friendTree1.nominalWeight * friendTree2.correction" }
```

With `treeList`, `name` identifies the group for configuration overrides, and each tree name becomes its ROOT friend alias. `path` must contain only the file path, without a `:tree` suffix. The list must be nonempty and contain nonempty tree names. Grouped entries can be mixed with single-tree entries; all resulting aliases must be unique within each main-file entry. Without `treeList`, `name` remains the alias as before.

Both main-file and friend entries accept `isEnabled` (default: `true`). Disabled main files are excluded from the chain; disabled friends are not attached. For grouped friends, `isEnabled` applies to every tree in the group. Every enabled file entry must declare the same enabled friend aliases. Each enabled friend tree must have exactly the same number and order of entries as its corresponding main tree. A branch with a unique name may be referenced without its alias, but `friendAlias.branch` is the recommended form because it remains unambiguous.


#### data

All options from MC

| Option                   | Type   | Description                                                                                                                     | Default |
|--------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------|---------|
| name                     | string | Name of the data/toy entry                                                                                                      |         |
| fromModel                | bool   | Use config from model as a base for a data entry                                                                                | false   |
| useReweightEngine        | bool   | Force to load the dials while loading the data. Used for custom toys that use the reweight engine for generating the histograms | false   |
| evalModelAt              | json   | Specify which sets of parameters to eval the model and copy the events from. Parameter injector format                          |         |
| overridePropagatorConfig | json   | Use a custom set of config option that will override the propagator config. Any parameter can be changed from here              |         |
