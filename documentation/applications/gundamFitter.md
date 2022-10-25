## gundamFitter

### Description

The `gundamFitter` app is an interface to the `FitterEngine` which is in charge
of setting up datasets, samples, fit parameters according to a provided config
file.

The list of options can be displayed by running `gundamFitter` without command
line arguments. The program should immediately stop after complaining that no
config file was provided.

### Config options

| Option                                                 | Type         | Description                                                     | Default |
|--------------------------------------------------------|--------------|-----------------------------------------------------------------|---------|
| [fitterEngineConfig](../configuration/FitterEngine.md) | json         | FitterEngine config                                             |         |
| minGundamVersion                                       | string       | gundamFitter will stop if the version is lower                  |         |
| outputFolder                                           | string       | Folder where the output file is written                         | ./      |
| fit (--dry-run)                                        | bool         | Run the fit once everything is loaded                           | true    |
| generateSamplePlots                                    | bool         | Draw sample histograms according to the PlotGenerator config    | true    |
| generateOneSigmaPlots (--one-sigma)                    | bool         | Draw comparison plots when each parameter is pushed to +1 sigma | false   |
| allParamVariations                                     | list(double) | List of points to perform individual parameter variation        |         |

