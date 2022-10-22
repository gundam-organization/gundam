## FitParameter

[< back to parent (FitParameterSet)](./FitParameterSet.md)

### Description

### Config options

| Option                             | Type         | Description                               | Default    |
|------------------------------------|--------------|-------------------------------------------|------------|
| parameterName                      | string       | define parameter with name                |            |
| parameterIndex                     | int          | define parameter with index               |            |
| isEnabled                          | bool         | use parameter in the propagator           | true       |
| priorType                          | string       | penalty LLH shape                         | Gaussian   |
| priorValue                         | double       | prior value                               | nan        |
| parameterLimits                    | pair(double) | min max                                   | [nan, nan] |
| [dialSetDefinitions](./DialSet.md) | list(json)   | definition of dialsets for this parameter |            |
