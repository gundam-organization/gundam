### Description

The `gundamFitter` app is an interface to the `FitterEngine` which is in charge
of setting up datasets, samples, fit parameters according to a provided config
file.

The list of options can be displayed by running `gundamFitter` without command
line arguments. The program should immediately stop after complaining that no
config file was provided.

### Usage

Most inputs are define in the  fitter config file and are called using the `-c` argument followed by the path to corresponding configuration file. 
```bash
gundamFitter -c path/to/config.yaml
```
Config files can be written in the `.yaml` format or `.json`. Alternatively one cans use [gundamConfigUnfolder](gundamConfigUnfolder.md) to convert `.yaml` inputs into one `.json`. 

The number of parallel threads used by the fit can be defined by using the `-t` argument as follows:
```bash
gundamFitter -c path/to/config.yaml -t 15
```

If not specified in the config files one can also provide the path to the output file with the `-o` argument:
```bash 
gundamFitter -c path/to/config.yaml -o path/to/output.root -t 15
```

Override files can be past as arguments with `-of` to override default config without having to modify it:
```bash
gundamFitter -c path/to/config.yaml -o path/to/output.root -of path/to/override.yaml -t 15
```

### Trigger options

A certain number of trigger options can be used depending  on what the user needs:

| Option           | Description                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------- |
| -d               | Perform the full sequence of initialization, but don't do the actual fit.                |
| -a               | Use MC dataset to fill the data histograms                                               |
| --pca            | Enable principle component analysis for eigen decomposed parameter sets                  |
| --skip-hesse     | Don't perform postfit error evaluation                                                   |
| --skip-simplex   | Don't run SIMPLEX before the actual fit                                                  |
| --kick-mc        | Push MC parameters away from their prior to help the fit converge                        |
| --one-sigma      | Generate one sigma plots                                                                 |
| --light-mode     | Disable plot generation                                                                  |
| --no-dial-cache  | Disable cache handling for dial eval                                                     |
| --ignore-version | Don't check GUNDAM version with config request                                           |
| --scan           | Enable parameter scan before and after the fit (can provide nSteps)                      |
| ---scan-line     | Provide par injector files: start and end point or only end point (start will be prefit) |
| --toy            | Run a toy fit (optional arg to provide toy index)                                        |

For a complete list of options run command without arguments.

### Config options

| Option                                                 | Type   | Description                                    | Default |
| ------------------------------------------------------ | ------ | ---------------------------------------------- | ------- |
| [fitterEngineConfig](../configuration/FitterEngine.md) | json   | FitterEngine config                            |         |
| minGundamVersion                                       | string | gundamFitter will stop if the version is lower |         |
| outputFolder                                           | string | Folder where the output file is written        | ./      |

