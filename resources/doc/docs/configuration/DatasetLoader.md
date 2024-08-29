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

| Option                  | Type                | Description                                                     | Default |
|-------------------------|---------------------|-----------------------------------------------------------------|---------|
| tree                    | string              | Name of the TTree containing the data in each file              |         |
| selectionCutFormula     | string              | Global selection cut (should return 0 if not selected)          |         |
| nominalWeightFormula    | string/list(string) | Formula that returns the base weight of a given event           |         |
| filePathList            | list(string)        | list of ROOT files containing the TTree                         |         |
| additionalLeavesStorage | list(string)        | list of variables to be stored in memory                        |         |
| variablesTransform      | list(json)          | list of transform operations that will be applied while loading |         |
| variableDict        | list(json)          | dictionary translating a leaf/formula to variable name          |         |
| fromHistContent         | json                | use hist bin content directly. This will create dummy events    |         |


#### data

All options from MC

| Option | Type   | Description                                                               | Default |
|--------|--------|---------------------------------------------------------------------------|---------|
| name   | string | Name of the data/toy entry                                                |         |
| fromMc | bool   | Inherit all parameter from MC. All other entries are treated as overrides | false   |

