---
layout: default
title: gundamFitter
---

### Description

The `gundamFitter` app is an interface to the `FitterEngine` which is in charge
of setting up datasets, samples, fit parameters according to a provided config
file.

The list of options can be displayed by running `gundamFitter` without command
line arguments. The program should immediately stop after complaining that no
config file was provided.

### Usage

Most inputs are defined in the  fitter config file and are called using the `-c` argument followed by the path to corresponding configuration file. 
```bash
gundamFitter -c path/to/config.yaml
```
Config files can be written in the `.yaml` format or `.json`. Alternatively one can use [gundamConfigUnfolder](gundamConfigUnfolder.md) to convert `.yaml` inputs into `.json`. 

The number of parallel threads used by the fit can be defined by using the `-t` argument as follows:
```bash
gundamFitter -c path/to/config.yaml -t 15
```

If not specified in the config files one can also provide the path to the output file with the `-o` argument:
```bash 
gundamFitter -c path/to/config.yaml -o path/to/output.root -t 15
```

Override files can be passed as arguments with `-of` to override the default config without having to modify it:
```bash
gundamFitter -c path/to/config.yaml -o path/to/output.root -of path/to/override.yaml -t 15
```

### Trigger options

A certain number of trigger options can be used depending  on what the user needs:

| Option | Description |
| ------ | ----------- |
| `-c`, `--config-file` | Specify path to the fitter config file |
| `-of`, `--override-files` | Provide config files that will override keys |
| `-O`, `--override` | Add a command line override [e.g. `/fitterEngineConfig/engineType=mcmc`] |

| Runtime Options           | Description                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------- |
| --pca            | Enable principle component analysis for eigen decomposed parameter sets                  |
| -t, --nb-threads         | Specify nb of parallel threads                 |
| -s, --seed         | Set random seed              |

| Fit Options           | Description                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------- |
| -a, --asimov               | Use MC dataset to fill the data histograms                                               |
| --skip-hesse     | Don't perform postfit error evaluation                                                   |
| --toy            | Run a toy fit (optional arg to provide toy index)                                        |
| --inject-parameters           | Inject parameters defined in the provided config file                                       |

  | Output Options        | Description                                                                              |
  | --------------------- | ---------------------------------------------------------------------------------------- |
  | -o, --out-file        | Specify the output file                                                                  |
  | --out-dir             | Specify the output directory                                                             |
  | --appendix            | Add appendix to the output file name                                                     |
  | --scan                | Enable parameter scan before and after the fit (can provide nSteps)                     |
  | --scan-line           | Provide par injector files: start and end point or only end point (start will be prefit) |
  | --one-sigma           | Generate one sigma plots                                                                 |
  | --light-mode          | Disable plot 
  generation                                                                  |
  
| Debug Options                | Description                                                                                   |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| -d, --dry-run                | Perform the full sequence of initialization, but don't do the actual fit.                    |
| -dd, --super-dry-run         | Only reads the config files.                                                                 |
| -ddd, --hyper-dry-run        | Only unfolds the config files.                                                               |
| --debug                      | Enable debug verbose (can provide verbose level arg)                                          |
| --kick-mc                    | Amount to push the starting parameters away from their prior values (default: 0)             |
| --ignore-version             | Don't check GUNDAM version with config request                                                |
| -me, --max-events            | Set the maximum number of events to load per dataset                                          |
| -fe, --fraction-of-entries   | Set the fraction of the total entries of each TTree that will be read                         |

### Additional Options

| Option                       | Description                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| --skip-simplex               | Don't run SIMPLEX before the actual fit                                                  |
| --one-sigma                  | Generate one sigma plots                                                                 |
| --light-mode                 | Disable plot generation                                                                  |
| --no-dial-cache              | Disable cache handling for dial eval                                                     |



For a complete list of options run command without arguments.

### Config options

| Option                                                 | Type   | Description                                    | Default |
| ------------------------------------------------------ | ------ | ---------------------------------------------- | ------- |
| [fitterEngineConfig](../configuration/FitterEngine.md) | json   | FitterEngine config                            |         |
| minGundamVersion                                       | string | gundamFitter will stop if the version is lower |         |
| outputFolder                                           | string | Folder where the output file is written        | ./      |
