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
| devSingleThreadEventSelection        | bool   | Force the event selection to be performed in single thread          | false   |
| devSingleThreadEventLoaderAndIndexer | bool   | Force the event loading to be performed in single thread            | false   |


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
    friendList:
      - name: "weights"
        path: "${DATA_DIR}/weights-run2.root:weights"
      - name: "truth"
        path: "${DATA_DIR}/truth-run2.root:truth"

variableDict:
  - { name: "weight", expr: "weights.nominalWeight" }
```

Every named file entry must declare the same friend aliases. Each friend tree must have exactly the same number and order of entries as its corresponding main tree. A branch with a unique name may be referenced without its alias, but `friendAlias.branch` is the recommended form because it remains unambiguous.


#### data

All options from MC

| Option                   | Type   | Description                                                                                                                     | Default |
|--------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------|---------|
| name                     | string | Name of the data/toy entry                                                                                                      |         |
| fromModel                | bool   | Use config from model as a base for a data entry                                                                                | false   |
| useReweightEngine        | bool   | Force to load the dials while loading the data. Used for custom toys that use the reweight engine for generating the histograms | false   |
| evalModelAt              | json   | Specify which sets of parameters to eval the model and copy the events from. Parameter injector format                          |         |
| overridePropagatorConfig | json   | Use a custom set of config option that will override the propagator config. Any parameter can be changed from here              |         |
