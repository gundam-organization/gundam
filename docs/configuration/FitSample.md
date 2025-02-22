## Sample

[< back to parent (SampleSet)](./SampleSet.md)

### Description

### Config options


| Option        | Type         | Description                               | Default |
|---------------|--------------|-------------------------------------------|---------|
| **name**      | string       | sample name                               |         |
| isEnabled     | bool         | include the sample in the propagator      | true    |
| binning       | string       | path to binning definition file           |         |
| selectionCuts | string       | tree selection formula                    |         |
| datasets      | list(string) | list of dataset will populate this sample | `{"*"}` |
| mcNorm        | double       | MC sample global normalisation            | 1       |
| dataNorm      | double       | data sample global normalisation          | 1       |

