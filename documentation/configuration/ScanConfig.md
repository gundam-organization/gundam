## ScanConfig

[< back to parent (FitterEngine)](./FitterEngine.md)


#### Scan options

| scanConfig Options  | Type         | Description                                         | Default |
|---------------------|--------------|-----------------------------------------------------|---------|
| nbPoints            | int          | N sample point of LLH scan                          | 100     |
| parameterSigmaRange | list(double) | Scan around the current point at +/- X prior sigmas | {-3, 3} |
| varsConfig          | json         | List of quantities to scan                          |         |
| useParameterLimits  | bool         | Don't scan LLH out of bounds                        | true    |


#### Scan/Vars options

| varsConfig Options     | Type | Description                                | Default |
|------------------------|------|--------------------------------------------|---------|
| llh                    | bool | Save total LLH scans                       | true    |
| llhStat                | bool | Save stat. LLH scans                       | true    |
| llhPenalty             | bool | Save penalty LLH scans                     | true    |
| llhStatPerSample       | bool | Save stat. LLH per sample scans            | false   |
| llhStatPerSamplePerBin | bool | Save stat. LLH per sample per bin scans    | false   |
| weightPerSample        | bool | Save total weight per sample scans         | false   |
| weightPerSamplePerBin  | bool | Save total weight per sample per bin scans | false   |
